from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from mcq_inference.common import MCQPrediction, render_array_to_png, save_bytes_as_supported_image


ANSWER_RE = re.compile(r"\b([A-Z])\b")
MULTI_LETTER_RE = re.compile(r"[A-Z]")


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open() as handle:
        return [json.loads(line) for line in handle]


def _render_inline_options_block(options: dict[str, str]) -> str:
    return " ".join(f"({letter}) {text}" for letter, text in sorted(options.items()))


def build_prompt(
    record: dict[str, Any],
    prompt_variant: str = "normal",
    *,
    duplicate_options_in_question: bool = False,
) -> tuple[str, str]:
    if prompt_variant not in {"normal", "text_only", "options_only", "image_options_only"}:
        raise ValueError(f"Unsupported prompt_variant: {prompt_variant}")
    metadata = record.get("metadata") or {}
    response_type = str(metadata.get("response_type") or "single_choice").strip().lower()
    raw_prompt_includes_options = bool(metadata.get("raw_prompt_includes_options"))
    options_block = "\n".join(f"{letter}. {text}" for letter, text in sorted(record["options"].items()))
    if prompt_variant == "image_options_only":
        evidence_instruction = "Use the provided images and answer choices."
    elif prompt_variant == "options_only":
        evidence_instruction = "Use only the provided answer choices."
    elif prompt_variant == "text_only":
        evidence_instruction = "Use the provided text and answer choices."
    else:
        evidence_instruction = "Use all provided images and the full text."

    if response_type == "multi_letter_csv":
        system_msg = (
            "You are a medical vision-language assistant solving multiple-choice questions. "
            f"{evidence_instruction} "
            "Return JSON with keys `answer` and `rationale`. "
            "`answer` must be a comma-separated sequence of uppercase option letters, one per step, "
            "for example `A,C,B,A`."
        )
        if prompt_variant in {"options_only", "image_options_only"}:
            option_blocks = metadata.get("joint_option_blocks") or []
            rendered_blocks = []
            for idx, block in enumerate(option_blocks, start=1):
                rendered_blocks.append(f"Step {idx} options:\n" + "\n".join(str(item) for item in block))
            user_msg = "\n\n".join(rendered_blocks) + "\n\nReturn JSON only."
            return system_msg, user_msg
        if raw_prompt_includes_options:
            user_msg = f"{record['question']}\n\nReturn JSON only."
            return system_msg, user_msg
        user_msg = (
            f"Question:\n{record['question']}\n\n"
            f"Options:\n{options_block}\n\n"
            "Return JSON only."
        )
        return system_msg, user_msg

    system_msg = (
        "You are a medical vision-language assistant solving multiple-choice questions. "
        f"{evidence_instruction} "
        "Return JSON with keys `answer` and `rationale`. "
        "`answer` must be exactly one uppercase option letter from the provided choices."
    )
    if prompt_variant in {"options_only", "image_options_only"}:
        user_msg = (
            f"Options:\n{options_block}\n\n"
            "Return JSON only."
        )
        return system_msg, user_msg

    question_text = str(record["question"])
    if duplicate_options_in_question:
        question_text = f"{question_text}\n\n{_render_inline_options_block(record['options'])}"

    user_msg = (
        f"Question:\n{question_text}\n\n"
        f"Options:\n{options_block}\n\n"
        "Return JSON only."
    )
    return system_msg, user_msg


def _resolve_manifest_path(sample_root: Path, raw_path: str) -> str:
    path = Path(raw_path)
    if not path.is_absolute():
        path = sample_root / path
    return str(path.resolve())


def _resolve_image_refs(
    record: dict[str, Any],
    *,
    sample_root: Path,
    cache_root: Path,
) -> list[str]:
    resolved: list[str] = []
    for image_ref in record.get("image_refs", []):
        kind = image_ref["kind"]
        if kind == "file":
            resolved.append(_resolve_manifest_path(sample_root, str(image_ref["path"])))
            continue

        if kind != "zip":
            raise ValueError(f"Unsupported image ref kind in manifest: {kind}")

        zip_path = Path(str(image_ref["zip_path"]))
        if not zip_path.is_absolute():
            zip_path = sample_root / zip_path
        member = str(image_ref["member"])
        cache_key = hashlib.sha1(
            json.dumps(
                {
                    "kind": "zip",
                    "zip_path": str(zip_path.resolve()),
                    "member": member,
                    "render_npy": bool(image_ref.get("render_npy")),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        base = cache_root / record["dataset"] / cache_key
        if image_ref.get("render_npy") or member.endswith(".npy"):
            destination = base.with_suffix(".png")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path) as archive:
                    raw = archive.read(member)
                render_array_to_png(raw, destination)
            resolved.append(str(destination))
            continue

        candidate_paths = list((base.parent).glob(f"{base.name}.*"))
        if candidate_paths:
            resolved.append(str(candidate_paths[0]))
            continue

        with zipfile.ZipFile(zip_path) as archive:
            raw = archive.read(member)
        resolved_path = save_bytes_as_supported_image(
            raw,
            base,
            preferred_suffix=Path(member).suffix or ".png",
        )
        resolved.append(str(resolved_path))

    return resolved


def create_model(
    model_name: str,
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    max_concurrency: int = 8,
    max_images: Optional[int] = None,
    vllm_max_model_len: Optional[int] = None,
    vllm_max_num_batched_tokens: Optional[int] = None,
    vllm_gpu_memory_utilization: Optional[float] = None,
    qwen35_mode: Optional[str] = None,
    gemma4_thinking: bool = False,
    vllm_enforce_eager: bool = False,
    vllm_compilation_mode: Optional[str] = None,
    vllm_cudagraph_mode: Optional[str] = None,
    vllm_enable_chunked_prefill: Optional[bool] = None,
    vllm_async_scheduling: Optional[bool] = None,
    vllm_enable_prefix_caching: Optional[bool] = None,
    vllm_mm_processor_cache_gb: Optional[float] = None,
    vllm_mm_processor_cache_type: Optional[str] = None,
):
    lower = model_name.lower()
    if lower.startswith("openai/") or lower.startswith("gpt-") or "/gpt-" in lower:
        from model.gpt import APIModel

        api_model_name = model_name.split("/", 1)[1] if lower.startswith("openai/") else model_name
        return APIModel(
            model_name=api_model_name,
            temperature=0.0 if temperature is None else temperature,
            max_tokens=max_tokens,
            max_concurrency=max_concurrency,
            reasoning_effort=None,
            verbosity=None,
            set_max_tokens=(max_tokens is not None),
            request_timeout_s=300.0,
        )

    from model.vllmModels import vllmModels

    return vllmModels(
        model_name=model_name,
        max_tokens=(6000 if max_tokens is None else max_tokens),
        temperature=0.6 if temperature is None else temperature,
        top_p=0.95 if top_p is None else float(top_p),
        top_k=-1 if top_k is None else int(top_k),
        min_p=0.0 if min_p is None else float(min_p),
        presence_penalty=0.0 if presence_penalty is None else float(presence_penalty),
        repetition_penalty=1.0 if repetition_penalty is None else float(repetition_penalty),
        max_images=(47 if max_images is None else int(max_images)),
        max_model_len=(
            None if vllm_max_model_len is None else int(vllm_max_model_len)
        ),
        max_num_batched_tokens=(
            16384 if vllm_max_num_batched_tokens is None else int(vllm_max_num_batched_tokens)
        ),
        gpu_memory_utilization=(
            0.95
            if vllm_gpu_memory_utilization is None
            else float(vllm_gpu_memory_utilization)
        ),
        qwen35_mode=qwen35_mode,
        gemma4_thinking=gemma4_thinking,
        enforce_eager=(True if vllm_enforce_eager else None),
        compilation_mode=vllm_compilation_mode,
        cudagraph_mode=vllm_cudagraph_mode,
        enable_chunked_prefill=vllm_enable_chunked_prefill,
        async_scheduling=vllm_async_scheduling,
        enable_prefix_caching=vllm_enable_prefix_caching,
        mm_processor_cache_gb=vllm_mm_processor_cache_gb,
        mm_processor_cache_type=vllm_mm_processor_cache_type,
    )


def canonicalize_multi_letter_answer(raw_text: Any, expected_count: Optional[int]) -> Optional[str]:
    text = str(raw_text or "").strip().upper()
    if not text:
        return None

    pieces = [piece.strip() for piece in re.split(r"[\s,\[\]\(\)\{\};:/|]+", text) if piece.strip()]
    letters = [piece for piece in pieces if len(piece) == 1 and piece.isalpha()]
    if not letters:
        letters = MULTI_LETTER_RE.findall(text)
    if expected_count is not None and len(letters) != int(expected_count):
        return None
    if not letters:
        return None
    return ",".join(letters)


def parse_prediction(
    raw_response: Any,
    options: dict[str, str],
    metadata: Optional[dict[str, Any]] = None,
) -> tuple[Optional[str], Optional[str], Any]:
    metadata = metadata or {}
    response_type = str(metadata.get("response_type") or "single_choice").strip().lower()
    expected_answer_count = metadata.get("expected_answer_count")

    if isinstance(raw_response, dict):
        answer = str(raw_response.get("answer", "")).strip().upper()
        rationale = str(raw_response.get("rationale", "")).strip() or None
        if response_type == "multi_letter_csv":
            normalized = canonicalize_multi_letter_answer(answer, expected_answer_count)
            if normalized:
                return normalized, rationale, raw_response
        elif answer in options:
            return answer, rationale, raw_response
        raw_text = json.dumps(raw_response, ensure_ascii=False)
    else:
        raw_text = str(raw_response or "").strip()

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            answer = str(parsed.get("answer", "")).strip().upper()
            rationale = str(parsed.get("rationale", "")).strip() or None
            if response_type == "multi_letter_csv":
                normalized = canonicalize_multi_letter_answer(answer, expected_answer_count)
                if normalized:
                    return normalized, rationale, parsed
            elif answer in options:
                return answer, rationale, parsed
    except Exception:
        parsed = raw_text

    if response_type == "multi_letter_csv":
        normalized = canonicalize_multi_letter_answer(raw_text, expected_answer_count)
        if normalized:
            return normalized, None, parsed
        return None, None, parsed

    match = re.search(r'"answer"\s*:\s*"([A-Z])"', raw_text)
    if match and match.group(1) in options:
        return match.group(1), None, parsed

    for match in ANSWER_RE.finditer(raw_text):
        if match.group(1) in options:
            return match.group(1), None, parsed

    normalized_raw = raw_text.strip().lower()
    for letter, option_text in options.items():
        if normalized_raw == option_text.strip().lower():
            return letter, None, parsed

    return None, None, parsed


def run_inference(
    *,
    manifest_path: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int = 1,
    max_examples: Optional[int] = None,
    dataset_filter: Optional[set[str]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    max_concurrency: int = 8,
    max_images: Optional[int] = None,
    vllm_max_model_len: Optional[int] = None,
    vllm_max_num_batched_tokens: Optional[int] = None,
    vllm_gpu_memory_utilization: Optional[float] = None,
    qwen35_mode: Optional[str] = None,
    gemma4_thinking: bool = False,
    vllm_enforce_eager: bool = False,
    vllm_compilation_mode: Optional[str] = None,
    vllm_cudagraph_mode: Optional[str] = None,
    vllm_enable_chunked_prefill: Optional[bool] = None,
    vllm_async_scheduling: Optional[bool] = None,
    vllm_enable_prefix_caching: Optional[bool] = None,
    vllm_mm_processor_cache_gb: Optional[float] = None,
    vllm_mm_processor_cache_type: Optional[str] = None,
    prompt_variant: str = "normal",
    duplicate_options_in_question: bool = False,
    request_chunk_size: Optional[int] = None,
    use_structured_output: bool = True,
    show_progress: bool = True,
) -> tuple[Path, Path]:
    sample_root = manifest_path.parent
    records = load_manifest(manifest_path)
    if dataset_filter:
        records = [record for record in records if record["dataset"] in dataset_filter]
    if max_examples is not None:
        records = records[: max_examples]

    model = create_model(
        model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        max_concurrency=max_concurrency,
        max_images=max_images,
        vllm_max_model_len=vllm_max_model_len,
        vllm_max_num_batched_tokens=vllm_max_num_batched_tokens,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        qwen35_mode=qwen35_mode,
        gemma4_thinking=gemma4_thinking,
        vllm_enforce_eager=vllm_enforce_eager,
        vllm_compilation_mode=vllm_compilation_mode,
        vllm_cudagraph_mode=vllm_cudagraph_mode,
        vllm_enable_chunked_prefill=vllm_enable_chunked_prefill,
        vllm_async_scheduling=vllm_async_scheduling,
        vllm_enable_prefix_caching=vllm_enable_prefix_caching,
        vllm_mm_processor_cache_gb=vllm_mm_processor_cache_gb,
        vllm_mm_processor_cache_type=vllm_mm_processor_cache_type,
    )

    image_cache_root = sample_root / "_resolved_image_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    generate_kwargs = {
        "show_progress": show_progress,
    }
    is_vllm_model = model.__class__.__name__ == "vllmModels"
    if use_structured_output:
        generate_kwargs["schema"] = MCQPrediction
    if not is_vllm_model:
        generate_kwargs["batch_size"] = batch_size
    uses_images = bool(
        prompt_variant in {"normal", "image_options_only"}
        and any(record.get("image_paths") or record.get("image_refs") for record in records)
    )
    requested_request_chunk_size = (
        max(1, int(request_chunk_size))
        if request_chunk_size is not None
        else None
    )
    effective_request_chunk_size = len(records) or 1
    if is_vllm_model and uses_images and requested_request_chunk_size is None:
        requested_request_chunk_size = 2000
    if is_vllm_model and requested_request_chunk_size is not None and len(records) > requested_request_chunk_size:
        effective_request_chunk_size = requested_request_chunk_size
    uses_conditional_chunking = bool(is_vllm_model and effective_request_chunk_size < len(records))

    summary: dict[str, Any] = {
        "model_name": model_name,
        "prompt_variant": prompt_variant,
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
        "temperature": float(temperature) if temperature is not None else None,
        "top_p": float(top_p) if top_p is not None else None,
        "top_k": int(top_k) if top_k is not None else None,
        "min_p": float(min_p) if min_p is not None else None,
        "presence_penalty": float(presence_penalty) if presence_penalty is not None else None,
        "repetition_penalty": float(repetition_penalty) if repetition_penalty is not None else None,
        "qwen35_mode": qwen35_mode,
        "use_structured_output": bool(use_structured_output),
        "vllm_max_model_len": int(vllm_max_model_len) if vllm_max_model_len is not None else None,
        "vllm_max_num_batched_tokens": (
            int(vllm_max_num_batched_tokens)
            if vllm_max_num_batched_tokens is not None
            else None
        ),
        "vllm_gpu_memory_utilization": (
            float(vllm_gpu_memory_utilization)
            if vllm_gpu_memory_utilization is not None
            else None
        ),
        "duplicate_options_in_question": bool(duplicate_options_in_question),
        "request_submission_mode": (
            "conditional_chunked_generate_calls"
            if uses_conditional_chunking
            else "all_requests_single_generate_call"
        ),
        "request_chunk_size": effective_request_chunk_size if uses_conditional_chunking else None,
        "request_chunk_threshold": requested_request_chunk_size if is_vllm_model else request_chunk_size,
        "uses_images": uses_images,
        "total_examples": len(records),
        "answered": 0,
        "correct": 0,
        "datasets": defaultdict(lambda: {"total": 0, "answered": 0, "correct": 0}),
    }

    running_summary = {
        **summary,
        "accuracy": 0.0,
        "datasets": {},
        "status": "running",
    }
    summary_path.write_text(
        json.dumps(running_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with results_path.open("w", encoding="utf-8") as handle:
        num_chunks = (
            (len(records) + effective_request_chunk_size - 1) // effective_request_chunk_size
            if records
            else 0
        )
        for chunk_index, chunk_start in enumerate(range(0, len(records), effective_request_chunk_size), start=1):
            chunk_records = records[chunk_start: chunk_start + effective_request_chunk_size]
            if num_chunks > 1:
                print(
                    f"Running generate chunk {chunk_index}/{num_chunks} "
                    f"with {len(chunk_records)} requests..."
                )
            prompts = []
            for record in chunk_records:
                system_msg, user_msg = build_prompt(
                    record,
                    prompt_variant=prompt_variant,
                    duplicate_options_in_question=duplicate_options_in_question,
                )
                if prompt_variant in {"text_only", "options_only"}:
                    image_paths = []
                elif record.get("image_paths"):
                    image_paths = [
                        _resolve_manifest_path(sample_root, str(image_path))
                        for image_path in record["image_paths"]
                    ]
                else:
                    image_paths = _resolve_image_refs(
                        record,
                        sample_root=sample_root,
                        cache_root=image_cache_root,
                    )
                prompts.append((system_msg, user_msg, image_paths))

            raw_outputs = model.generate(
                prompts,
                **generate_kwargs,
            )

            for record, output in zip(chunk_records, raw_outputs):
                raw_response, input_tokens, output_tokens = output[:3]
                predicted_answer, rationale, parsed_payload = parse_prediction(
                    raw_response,
                    record["options"],
                    metadata=record.get("metadata"),
                )
                is_correct = (
                    predicted_answer is not None
                    and record.get("correct_answer") is not None
                    and predicted_answer == record["correct_answer"]
                )

                dataset_stats = summary["datasets"][record["dataset"]]
                dataset_stats["total"] += 1
                summary["answered"] += int(predicted_answer is not None)
                summary["correct"] += int(is_correct)
                dataset_stats["answered"] += int(predicted_answer is not None)
                dataset_stats["correct"] += int(is_correct)

                result_record = {
                    "dataset": record["dataset"],
                    "example_id": record["example_id"],
                    "question": record["question"],
                    "options": record["options"],
                    "correct_answer": record.get("correct_answer"),
                    "predicted_answer": predicted_answer,
                    "predicted_answer_text": (
                        record["options"].get(predicted_answer)
                        if predicted_answer and predicted_answer in record["options"]
                        else None
                    ),
                    "rationale": rationale,
                    "is_correct": is_correct,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "raw_response": raw_response,
                    "parsed_payload": parsed_payload,
                    "image_paths": record.get("image_paths"),
                    "image_ref_count": len(record.get("image_refs", [])),
                    "metadata": record.get("metadata", {}),
                }
                handle.write(json.dumps(result_record, ensure_ascii=True) + "\n")

            handle.flush()
            os.fsync(handle.fileno())

            if num_chunks > 1:
                running_datasets = {
                    dataset: {
                        **stats,
                        "accuracy": (float(stats["correct"]) / float(stats["total"])) if stats["total"] else 0.0,
                    }
                    for dataset, stats in sorted(summary["datasets"].items())
                }
                running_summary = {
                    **summary,
                    "accuracy": (
                        float(summary["correct"]) / float(summary["total_examples"])
                        if summary["total_examples"]
                        else 0.0
                    ),
                    "datasets": running_datasets,
                    "status": "running",
                    "completed_chunks": chunk_index,
                    "total_chunks": num_chunks,
                }
                summary_path.write_text(
                    json.dumps(running_summary, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

    summary["accuracy"] = (
        float(summary["correct"]) / float(summary["total_examples"])
        if summary["total_examples"]
        else 0.0
    )
    summary["datasets"] = {
        dataset: {
            **stats,
            "accuracy": (float(stats["correct"]) / float(stats["total"])) if stats["total"] else 0.0,
        }
        for dataset, stats in sorted(summary["datasets"].items())
    }
    summary["status"] = "completed"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return results_path, summary_path
