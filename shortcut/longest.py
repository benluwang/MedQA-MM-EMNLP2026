#!/usr/bin/env python3
"""Length-gap shortcut selection and rewrite pipeline.

This script captures the length-gap intervention workflow without checking in
any datasets or generated JSON outputs.

The intended pipeline is:

1. select
   Find examples where the correct option is uniquely the longest option, rank
   them by the correct-vs-distractor length gap, and optionally apply externally
   supplied dataset filters or per-dataset caps.

2. make-batch
   Build OpenAI Batch API request JSONL asking a conservative medical editor to
   rewrite one or two distractors, leaving the answer key and correct option
   unchanged. Input images are attached when they are available from the
   manifest row and readable from the supplied image root.

3. materialize
   Parse batch responses and validate that each edit is safe. By default all
   valid rewrites are accepted after ranking by the original length gap; callers
   can pass explicit per-dataset caps when they need a fixed evaluation subset.
   An accepted edit must leave a distractor at least five visible characters
   longer than the correct option.

4. apply-edits
   Emit paired original and edited manifests for downstream evaluation.

All paths are command-line arguments. The code intentionally avoids hardcoded
local project paths, result file paths, API keys, and model outputs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_SETTING = "normal"
DEFAULT_ACCEPT_MIN_MARGIN_CHARS = 5

SYSTEM_MESSAGE = (
    "You are a conservative medical multiple-choice benchmark editor. You will "
    "receive a medical multimodal MCQ, all available images, the known correct "
    "answer, and a detected option-length shortcut. Your job is to remove the "
    "length shortcut only when it is safe, while preserving the intended medical "
    "meaning, image-grounded reasoning, answer key, option order, and number of "
    "options. Prefer no_safe_edit over a rewrite that changes medical truth "
    "conditions."
)

USER_TEMPLATE = """Please review this medical multimodal MCQ and, only if safe, minimally rewrite 1-2 distractor options to neutralize a length shortcut.

Context:
- The correct option is uniquely the longest option by visible character count.
- The goal is to make at least one distractor clearly longer than the correct option, so the correct option is no longer uniquely longest.
- Automated validation is strict. At least one revised distractor should be at least 5 visible characters longer than the correct option.
- You may change only 1 or 2 distractor options. Do not change the correct option.
- Use the images to understand the intended case before editing.

Hard rules:
- Do NOT change the question stem.
- Do NOT change the correct option text or answer label.
- Do NOT change option order, option letters, or the number of options.
- Do NOT edit more than two distractors.
- Preserve each edited distractor's original wrongness and answer type.
- Do not introduce a new diagnosis, organism, anatomy, laterality, severity, numeric value, treatment direction, causal claim, or negation.
- Do not make a distractor synonymous with the correct option.
- Do not duplicate or near-duplicate another option.
- Do not add generic filler such as "as clinically appropriate", "based on the image", or "in this patient" just to inflate length.
- Good edits include spelling out an abbreviation, making a distractor parallel to the option style, or a same-meaning paraphrase with explicit already-implied wording.
- If no safe edit can make any distractor at least 5 visible characters longer than the correct option, return no_safe_edit.
- If an edit may be valid but needs clinician/manual review, return needs_human_review.
- For no_safe_edit or needs_human_review, changed_letters and option_patches must be empty.
- Return JSON only.

Dataset: {dataset}
Example id: {example_id}
Correct answer: {correct_label}. {correct_text}

Length diagnostics:
Correct option length: {correct_chars} chars, {correct_words} words
Longest distractor before edit: {max_distractor_chars} chars
Gap to close: {gap_chars} chars
Minimum revised distractor length requested: {target_distractor_chars} chars

Recommended distractor candidates to consider first:
{candidate_text}

Question:
{question}

Options:
{options_text}

Useful metadata, if any:
{metadata_text}
"""

REWRITE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status",
        "changed_letters",
        "option_patches",
        "item_understanding",
        "chosen_edit_strategy",
        "length_shortcut_reduction",
        "medical_equivalence_rationale",
        "new_bias_risk_check",
        "residual_risk",
    ],
    "properties": {
        "status": {"type": "string", "enum": ["rewritten", "no_safe_edit", "needs_human_review"]},
        "changed_letters": {"type": "array", "items": {"type": "string"}},
        "option_patches": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["letter", "before", "after", "edit_type", "semantic_safety_note"],
                "properties": {
                    "letter": {"type": "string"},
                    "before": {"type": "string"},
                    "after": {"type": "string"},
                    "edit_type": {
                        "type": "string",
                        "enum": [
                            "abbreviation_expansion",
                            "parallel_detail",
                            "same_meaning_paraphrase",
                            "clarifying_parenthetical",
                            "other",
                        ],
                    },
                    "semantic_safety_note": {"type": "string"},
                },
            },
        },
        "item_understanding": {"type": "string"},
        "chosen_edit_strategy": {"type": "string"},
        "length_shortcut_reduction": {"type": "string"},
        "medical_equivalence_rationale": {"type": "string"},
        "new_bias_risk_check": {"type": "string"},
        "residual_risk": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}


@dataclass(frozen=True)
class CaseKey:
    dataset: str
    example_id: str
    option_signature: str

    def as_feature_key(self) -> str:
        return "|".join([self.dataset, self.example_id, self.option_signature])


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_label(value: Any) -> str | None:
    text = clean_space(value).upper()
    if not text:
        return None
    if text in LETTERS:
        return text
    match = re.match(r"^\(?\s*([A-Z])\s*\)?(?:[\.:\-]|$)", text)
    return match.group(1) if match else None


def normalize_example_id(dataset: str, example_id: Any) -> str:
    value = clean_space(example_id)
    if dataset == "Amboss":
        value = value.replace("/", "_")
    if dataset == "NEJM" and value.lower().startswith("nejm_"):
        value = value.split("_", 1)[1]
    return value


def normalize_options(raw_options: Any) -> list[tuple[str, str]]:
    if isinstance(raw_options, dict):
        pairs = []
        for key, value in raw_options.items():
            label = normalize_label(key) or clean_space(key).upper()
            if label:
                pairs.append((label, clean_space(value)))
        return sorted(pairs, key=lambda item: LETTERS.index(item[0]) if item[0] in LETTERS else 99)
    if isinstance(raw_options, list):
        return [(LETTERS[idx], clean_space(value)) for idx, value in enumerate(raw_options) if idx < len(LETTERS)]
    return []


def options_dict(raw_options: Any) -> dict[str, str]:
    return dict(normalize_options(raw_options))


def option_signature(options: Sequence[tuple[str, str]]) -> str:
    payload = json.dumps(list(options), ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def option_lengths(options: Sequence[tuple[str, str]]) -> dict[str, dict[str, int]]:
    out = {}
    for label, text in options:
        cleaned = clean_space(text)
        out[label] = {"chars": len(cleaned), "words": len(re.findall(r"\S+", cleaned))}
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_counts(text: str | None) -> dict[str, int]:
    if not text:
        return {}
    out: dict[str, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = int(value.strip())
    return out


def parse_dataset_filter(text: str | None) -> set[str] | None:
    if not text:
        return None
    datasets = {part.strip() for part in text.split(",") if part.strip()}
    return datasets or None


def parse_manifest_args(values: Sequence[str] | None, map_path: Path | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if map_path:
        payload = json.loads(map_path.read_text(encoding="utf-8"))
        for key, value in payload.items():
            out[str(key)] = Path(value)
    for value in values or []:
        dataset, path = value.split("=", 1)
        out[dataset.strip()] = Path(path.strip())
    return out


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else base_dir / path


def record_dataset(row: dict[str, Any], fallback: str | None = None) -> str:
    return clean_space(row.get("dataset") or row.get("dataset_key") or row.get("_dataset_label") or fallback)


def row_matches_dataset(row: dict[str, Any], dataset: str) -> bool:
    got = record_dataset(row)
    return not got or got == dataset


def feature_key_for_row(dataset: str, row: dict[str, Any]) -> str:
    opts = normalize_options(row.get("options"))
    key = CaseKey(
        dataset=dataset,
        example_id=normalize_example_id(dataset, row.get("example_id")),
        option_signature=option_signature(opts),
    )
    return key.as_feature_key()


def load_manifest_records(manifest_paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for dataset, path in manifest_paths.items():
        for row in read_jsonl(path):
            if not row_matches_dataset(row, dataset):
                continue
            key = feature_key_for_row(dataset, row)
            enriched = dict(row)
            enriched["_dataset_label"] = dataset
            by_key[key] = enriched
    return by_key


def load_examples_from_accuracy_json(
    accuracy_json: Path,
    *,
    setting: str,
    source_root: Path,
) -> dict[str, dict[str, Any]]:
    payload = json.loads(accuracy_json.read_text(encoding="utf-8"))
    examples: dict[str, dict[str, Any]] = {}
    source_paths = []
    for rec in payload.get("flat_records", []):
        if rec.get("setting_key") != setting:
            continue
        dataset = clean_space(rec.get("dataset") or rec.get("dataset_key"))
        source = rec.get("source_predictions")
        if not dataset or not source:
            continue
        source_paths.append((dataset, resolve_path(str(source), source_root)))

    seen_sources: set[tuple[str, str]] = set()
    for dataset, path in source_paths:
        source_key = (dataset, str(path))
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        for row in read_jsonl(path):
            if not row_matches_dataset(row, dataset):
                continue
            opts = normalize_options(row.get("options"))
            if not opts:
                continue
            feature_key = feature_key_for_row(dataset, row)
            if feature_key not in examples:
                examples[feature_key] = {
                    "feature_key": feature_key,
                    "dataset": dataset,
                    "example_id": normalize_example_id(dataset, row.get("example_id")),
                    "question": clean_space(row.get("question")),
                    "options": opts,
                    "correct_answer": normalize_label(row.get("correct_answer")),
                    "record": dict(row),
                }
            else:
                existing = examples[feature_key]
                if len(clean_space(row.get("question"))) > len(existing.get("question", "")):
                    existing["question"] = clean_space(row.get("question"))
                    existing["record"] = dict(row)
    return examples


def build_selected_cases(
    examples: dict[str, dict[str, Any]],
    manifest_records: dict[str, dict[str, Any]],
    *,
    selection_counts: dict[str, int] | None = None,
    datasets: set[str] | None = None,
) -> list[dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature_key, ex in examples.items():
        dataset = ex["dataset"]
        if datasets is not None and dataset not in datasets:
            continue
        if selection_counts and dataset not in selection_counts:
            continue
        options = list(ex["options"])
        lengths = option_lengths(options)
        chars = {label: vals["chars"] for label, vals in lengths.items()}
        words = {label: vals["words"] for label, vals in lengths.items()}
        gold = ex.get("correct_answer")
        if not gold or gold not in chars or len(chars) < 2:
            continue
        max_chars = max(chars.values())
        max_labels = [label for label, value in chars.items() if value == max_chars]
        if chars[gold] != max_chars or len(max_labels) != 1:
            continue
        distractor_max_chars = max(value for label, value in chars.items() if label != gold)
        distractor_max_words = max(value for label, value in words.items() if label != gold)
        record = manifest_records.get(feature_key) or ex.get("record") or {}
        option_map = options_dict(record.get("options")) or dict(options)
        pools[dataset].append(
            {
                "feature_key": feature_key,
                "dataset": dataset,
                "example_id": ex["example_id"],
                "gap_chars": chars[gold] - distractor_max_chars,
                "gap_words": words[gold] - distractor_max_words,
                "correct_chars": chars[gold],
                "correct_words": words[gold],
                "max_distractor_chars": distractor_max_chars,
                "correct_answer": gold,
                "correct_text": option_map.get(gold, dict(options).get(gold, "")),
                "options": option_map,
                "record": record,
            }
        )

    selected: list[dict[str, Any]] = []
    for dataset in sorted(pools):
        ranked = sorted(
            pools.get(dataset, []),
            key=lambda row: (row["gap_chars"], row["gap_words"], row["example_id"]),
            reverse=True,
        )
        if selection_counts and dataset in selection_counts:
            ranked = ranked[: selection_counts[dataset]]
        selected.extend(ranked)
    selected.sort(key=lambda row: (row["dataset"], -float(row["gap_chars"]), str(row["example_id"])))
    return selected


def candidate_text(case: dict[str, Any]) -> str:
    lengths = option_lengths(sorted(case["options"].items()))
    rows = []
    gold = case["correct_answer"]
    for label, text in sorted(case["options"].items()):
        if label == gold:
            continue
        needed = max(5, int(case["correct_chars"] - lengths[label]["chars"] + DEFAULT_ACCEPT_MIN_MARGIN_CHARS))
        rows.append(
            {
                "label": label,
                "chars": lengths[label]["chars"],
                "words": lengths[label]["words"],
                "needed": needed,
                "text": text,
            }
        )
    rows.sort(key=lambda row: (row["needed"], -row["chars"]))
    return "\n".join(
        f"- {row['label']}: {row['chars']} chars, needs about +{row['needed']} chars for a 5-char safety buffer: {row['text']}"
        for row in rows[:4]
    )


def compact_metadata(meta: Any, max_chars: int = 7000) -> str:
    if not isinstance(meta, dict) or not meta:
        return "(none)"
    keep = [
        "question_stem",
        "raw_hint",
        "raw_correct_answer",
        "option_explanations",
        "option_percent",
        "difficulty",
        "source_dataset",
        "manifest_scheme",
        "image_filenames",
    ]
    slim = {key: meta.get(key) for key in keep if key in meta}
    if not slim:
        slim = {key: meta[key] for key in sorted(meta)[:20]}
    text = json.dumps(slim, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def build_user_text(case: dict[str, Any]) -> str:
    record = case.get("record") or {}
    question = clean_space(record.get("question") or case.get("question"))
    options_text = "\n".join(f"{label}. {text}" for label, text in sorted(case["options"].items()))
    return USER_TEMPLATE.format(
        dataset=case["dataset"],
        example_id=case["example_id"],
        correct_label=case["correct_answer"],
        correct_text=case["correct_text"],
        correct_chars=int(case["correct_chars"]),
        correct_words=int(case["correct_words"]),
        max_distractor_chars=int(case["max_distractor_chars"]),
        gap_chars=int(case["gap_chars"]),
        target_distractor_chars=int(case["correct_chars"] + DEFAULT_ACCEPT_MIN_MARGIN_CHARS),
        candidate_text=candidate_text(case),
        question=question,
        options_text=options_text,
        metadata_text=compact_metadata(record.get("metadata")),
    )


def collect_image_candidates(record: dict[str, Any], image_root: Path | None) -> list[str]:
    raw_values: list[Any] = []
    for key in ("image", "image_path", "images", "image_paths", "image_filenames"):
        if key in record:
            raw_values.append(record[key])
    meta = record.get("metadata")
    if isinstance(meta, dict):
        for key in ("image", "image_path", "images", "image_paths", "image_filenames"):
            if key in meta:
                raw_values.append(meta[key])

    flat: list[str] = []
    for value in raw_values:
        if isinstance(value, str):
            flat.append(value)
        elif isinstance(value, list):
            flat.extend(str(item) for item in value if item)

    out: list[str] = []
    for value in flat:
        value = clean_space(value)
        if not value:
            continue
        if re.match(r"^https?://", value) or value.startswith("data:image/"):
            out.append(value)
            continue
        path = Path(value)
        if not path.is_absolute() and image_root is not None:
            path = image_root / path
        if path.exists():
            out.append(str(path))
    return list(dict.fromkeys(out))


def image_part(image_ref: str) -> dict[str, Any] | None:
    if re.match(r"^https?://", image_ref) or image_ref.startswith("data:image/"):
        return {"type": "input_image", "image_url": image_ref}
    path = Path(image_ref)
    if not path.exists():
        return None
    mime, _ = mimetypes.guess_type(path.name)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"}


def make_response_body(model: str, user_text: str, image_refs: Sequence[str], max_output_tokens: int) -> dict[str, Any]:
    user_content = [{"type": "input_text", "text": user_text}]
    for image_ref in image_refs:
        part = image_part(image_ref)
        if part is not None:
            user_content.append(part)
    return {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": user_content},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "length_gap_rewrite",
                "schema": REWRITE_JSON_SCHEMA,
                "strict": True,
            }
        },
        "max_output_tokens": max_output_tokens,
    }


def extract_output_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"].strip()
    parts: list[str] = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "input_text", "text"}:
                parts.append(str(content.get("text") or ""))
    if parts:
        return "".join(parts).strip()
    choices = body.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
    return ""


def parse_json_text(text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = text.strip()
    if not text:
        return None, "empty output"
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"json parse error: {exc}"
    if not isinstance(parsed, dict):
        return None, "parsed output is not an object"
    return parsed, None


def final_options_after(case: dict[str, Any], parsed: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    options = dict(case["options"])
    gold = case["correct_answer"]
    status = parsed.get("status")
    changed_letters = [clean_space(value).upper() for value in parsed.get("changed_letters") or []]
    patches = parsed.get("option_patches") or []
    if status == "rewritten":
        if not (1 <= len(patches) <= 2):
            errors.append("rewritten status must include 1-2 option patches")
        if len(changed_letters) != len(set(changed_letters)):
            errors.append("changed_letters contains duplicates")
        patched_labels: list[str] = []
        for patch in patches:
            if not isinstance(patch, dict):
                errors.append("option patch is not an object")
                continue
            label = clean_space(patch.get("letter")).upper()
            patched_labels.append(label)
            if label == gold:
                errors.append("attempted to change correct option")
            if label not in options:
                errors.append(f"unknown option label {label}")
                continue
            before = clean_space(patch.get("before"))
            after = clean_space(patch.get("after"))
            if before != options[label]:
                errors.append(f"before text mismatch for {label}")
            if not after:
                errors.append(f"empty after text for {label}")
            options[label] = after
        if sorted(changed_letters) != sorted(patched_labels):
            errors.append("changed_letters does not match option_patches")
    else:
        if changed_letters or patches:
            errors.append("non-rewritten status should not include patches")
    return options, errors


def validate_case(case: dict[str, Any], parsed: dict[str, Any] | None, raw_error: str | None) -> dict[str, Any]:
    if parsed is None:
        return {
            "validation_status": "api_or_parse_error",
            "validation_errors": [raw_error or "unknown error"],
            "final_options": case["options"],
            "target_met": False,
            "final_correct_unique_longest": True,
        }
    status = parsed.get("status")
    errors: list[str] = []
    if status not in {"rewritten", "no_safe_edit", "needs_human_review"}:
        errors.append(f"unknown status {status}")
    final_options, patch_errors = final_options_after(case, parsed)
    errors.extend(patch_errors)
    lengths = option_lengths(sorted(final_options.items()))
    gold = case["correct_answer"]
    correct_chars = lengths[gold]["chars"] if gold in lengths else 0
    distractor_lengths = [vals["chars"] for label, vals in lengths.items() if label != gold]
    max_distractor_chars = max(distractor_lengths) if distractor_lengths else 0
    max_len = max((vals["chars"] for vals in lengths.values()), default=0)
    max_labels = [label for label, vals in lengths.items() if vals["chars"] == max_len]
    if final_options.get(gold) != case["correct_text"]:
        errors.append("correct option text changed")
    if len(final_options) != len(case["options"]):
        errors.append("option count changed")
    duplicate_texts = [
        text
        for text, count in Counter(text.strip().lower() for text in final_options.values()).items()
        if count > 1
    ]
    if duplicate_texts:
        errors.append("duplicate option text after edit")
    target_met = max_distractor_chars > correct_chars
    final_correct_unique_longest = max_labels == [gold]
    if status == "rewritten" and not target_met:
        errors.append("rewritten output did not make a distractor longer than the correct option")
    if status == "rewritten" and errors:
        validation_status = "invalid_rewrite"
    elif status == "rewritten":
        validation_status = "valid_rewrite"
    elif errors:
        validation_status = "invalid_rewrite"
    else:
        validation_status = str(status)
    return {
        "validation_status": validation_status,
        "validation_errors": errors,
        "final_options": final_options,
        "final_correct_chars": correct_chars,
        "final_max_distractor_chars": max_distractor_chars,
        "final_correct_unique_longest": final_correct_unique_longest,
        "target_met": target_met,
        "changed_count": len(parsed.get("option_patches") or []),
    }


def batch_body_and_error(row: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    response = row.get("response") if isinstance(row.get("response"), dict) else {}
    body = response.get("body") or row.get("body") or {}
    error = row.get("error") or response.get("error")
    return body if isinstance(body, dict) else {}, error


def load_batch_results(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for row in read_jsonl(path):
        custom_id = row.get("custom_id")
        if custom_id:
            out[str(custom_id)] = row
    return out


def replace_options_like(raw_options: Any, final_options: dict[str, str]) -> Any:
    if isinstance(raw_options, list):
        return [final_options[label] for label in LETTERS[: len(raw_options)] if label in final_options]
    return {label: final_options[label] for label in sorted(final_options)}


def cmd_select(args: argparse.Namespace) -> None:
    manifest_paths = parse_manifest_args(args.manifest, args.manifest_map)
    examples = load_examples_from_accuracy_json(
        Path(args.accuracy_json),
        setting=args.setting,
        source_root=Path(args.source_root),
    )
    manifests = load_manifest_records(manifest_paths) if manifest_paths else {}
    selection_counts = parse_counts(args.selection_counts)
    datasets = parse_dataset_filter(args.datasets)
    selected = build_selected_cases(examples, manifests, selection_counts=selection_counts, datasets=datasets)
    write_jsonl(Path(args.output), selected)
    by_dataset = Counter(row["dataset"] for row in selected)
    print(json.dumps({"selected": len(selected), "by_dataset": dict(by_dataset)}, sort_keys=True))


def cmd_make_batch(args: argparse.Namespace) -> None:
    cases = read_jsonl(Path(args.selected_cases))
    rows = []
    image_root = Path(args.image_root) if args.image_root else None
    for index, case in enumerate(cases):
        user_text = build_user_text(case)
        image_refs = collect_image_candidates(case.get("record") or {}, image_root)
        if args.max_images >= 0:
            image_refs = image_refs[: args.max_images]
        custom_id = f"length_gap_{index:05d}_{case['dataset']}_{case['example_id']}"
        rows.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": make_response_body(args.model, user_text, image_refs, args.max_output_tokens),
                "metadata": {
                    "feature_key": case["feature_key"],
                    "dataset": case["dataset"],
                    "example_id": case["example_id"],
                    "correct_answer": case["correct_answer"],
                    "gap_chars": case["gap_chars"],
                    "gap_words": case["gap_words"],
                    "original_options": case["options"],
                    "image_count": len(image_refs),
                },
            }
        )
    write_jsonl(Path(args.output), rows)
    print(json.dumps({"batch_rows": len(rows), "output": args.output}, sort_keys=True))


def cmd_materialize(args: argparse.Namespace) -> None:
    cases = read_jsonl(Path(args.selected_cases))
    batch_rows = load_batch_results(Path(args.batch_output))
    accept_counts = parse_counts(args.accept_counts)
    min_margin = int(args.accept_min_margin_chars)
    result_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        custom_id = f"length_gap_{index:05d}_{case['dataset']}_{case['example_id']}"
        row = batch_rows.get(custom_id)
        parsed = None
        parse_error = "missing batch output row"
        raw_output_text = ""
        api_error = None
        if row:
            body, api_error = batch_body_and_error(row)
            raw_output_text = extract_output_text(body)
            parsed, parse_error = parse_json_text(raw_output_text)
            if api_error and parsed is None:
                parse_error = json.dumps(api_error, ensure_ascii=False)
        validation = validate_case(case, parsed, parse_error)
        result = {
            "custom_id": custom_id,
            "feature_key": case["feature_key"],
            "dataset": case["dataset"],
            "example_id": case["example_id"],
            "correct_answer": case["correct_answer"],
            "correct_text": case["correct_text"],
            "gap_chars": case["gap_chars"],
            "gap_words": case["gap_words"],
            "original_options": case["options"],
            "raw_output_text": raw_output_text,
            "parsed_output": parsed,
            "api_error": api_error,
            **validation,
        }
        if validation["validation_status"] in {"api_or_parse_error", "invalid_rewrite"}:
            errors.append(result)
        result_rows.append(result)

    result_rows.sort(key=lambda row: (str(row["dataset"]), -float(row.get("gap_chars") or 0), str(row["example_id"])))
    accepted_counts: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    nonaccepted: list[dict[str, Any]] = []
    for row in result_rows:
        dataset = str(row["dataset"])
        margin = float(row.get("final_max_distractor_chars") or 0) - float(row.get("final_correct_chars") or 0)
        row["final_length_margin_chars"] = margin
        if (
            row.get("validation_status") == "valid_rewrite"
            and margin >= min_margin
            and (not accept_counts or accepted_counts[dataset] < int(accept_counts.get(dataset, 0)))
        ):
            row["selection_status"] = "accepted"
            accepted_counts[dataset] += 1
            accepted.append(row)
        else:
            row["selection_status"] = "not_accepted"
            nonaccepted.append(row)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "rewrite_results.jsonl", result_rows)
    write_jsonl(output_dir / "rewrite_errors.jsonl", errors)
    write_jsonl(output_dir / "accepted_rewrites.jsonl", accepted)
    write_jsonl(output_dir / "nonaccepted_rewrites.jsonl", nonaccepted)
    write_jsonl(
        output_dir / "accepted_patch_manifest.jsonl",
        (
            {
                "feature_key": row["feature_key"],
                "dataset": row["dataset"],
                "example_id": row["example_id"],
                "correct_answer": row["correct_answer"],
                "gap_chars": row["gap_chars"],
                "original_options": row["original_options"],
                "final_options": row["final_options"],
                "parsed_output": row["parsed_output"],
            }
            for row in accepted
        ),
    )
    summary = {
        "rows": len(result_rows),
        "accepted": len(accepted),
        "accepted_by_dataset": dict(accepted_counts),
        "accept_targets": accept_counts,
        "accept_min_margin_chars": min_margin,
        "validation_status": dict(Counter(row["validation_status"] for row in result_rows)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def cmd_apply_edits(args: argparse.Namespace) -> None:
    manifest_paths = parse_manifest_args(args.manifest, args.manifest_map)
    accepted = read_jsonl(Path(args.accepted_rewrites))
    accepted_by_key = {row["feature_key"]: row for row in accepted}
    original_rows: list[dict[str, Any]] = []
    edited_rows: list[dict[str, Any]] = []
    accepted_order = {row["feature_key"]: idx for idx, row in enumerate(accepted)}
    for dataset, path in manifest_paths.items():
        for row in read_jsonl(path):
            key = feature_key_for_row(dataset, row)
            accepted_row = accepted_by_key.get(key)
            if not accepted_row:
                continue
            original = dict(row)
            edited = dict(row)
            edited["options"] = replace_options_like(row.get("options"), accepted_row["final_options"])
            metadata = edited.get("metadata") if isinstance(edited.get("metadata"), dict) else {}
            edited["metadata"] = dict(metadata)
            edited["metadata"]["length_gap_shortcut_edit"] = {
                "source": "length_gap_longest_pipeline",
                "correct_answer_unchanged": accepted_row["correct_answer"],
                "original_gap_chars": accepted_row["gap_chars"],
            }
            original["_length_gap_feature_key"] = key
            edited["_length_gap_feature_key"] = key
            original_rows.append(original)
            edited_rows.append(edited)
    original_rows.sort(key=lambda row: accepted_order.get(row.get("_length_gap_feature_key", ""), 10**9))
    edited_rows.sort(key=lambda row: accepted_order.get(row.get("_length_gap_feature_key", ""), 10**9))
    for row in original_rows + edited_rows:
        row.pop("_length_gap_feature_key", None)
    write_jsonl(Path(args.original_output), original_rows)
    write_jsonl(Path(args.edited_output), edited_rows)
    print(json.dumps({"original": len(original_rows), "edited": len(edited_rows)}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select", help="select uniquely-correct-longest length-gap candidates")
    select.add_argument("--accuracy-json", required=True)
    select.add_argument("--source-root", default=".", help="base directory for relative source_predictions paths")
    select.add_argument("--setting", default=DEFAULT_SETTING)
    select.add_argument("--manifest", action="append", help="dataset=manifest.jsonl; may be repeated")
    select.add_argument("--manifest-map", type=Path, help="JSON object mapping dataset labels to manifest paths")
    select.add_argument("--datasets", default=None, help="optional comma-separated dataset labels to include")
    select.add_argument("--selection-counts", default=None, help="optional comma list of per-dataset caps, formatted DATASET=N")
    select.add_argument("--output", required=True)
    select.set_defaults(func=cmd_select)

    make_batch = sub.add_parser("make-batch", help="write OpenAI Batch API request JSONL for selected cases")
    make_batch.add_argument("--selected-cases", required=True)
    make_batch.add_argument("--model", required=True)
    make_batch.add_argument("--output", required=True)
    make_batch.add_argument("--image-root", default=None)
    make_batch.add_argument("--max-images", type=int, default=-1, help="-1 means no explicit limit")
    make_batch.add_argument("--max-output-tokens", type=int, default=4096)
    make_batch.set_defaults(func=cmd_make_batch)

    materialize = sub.add_parser("materialize", help="validate batch outputs and accept safe edits")
    materialize.add_argument("--selected-cases", required=True)
    materialize.add_argument("--batch-output", required=True)
    materialize.add_argument("--output-dir", required=True)
    materialize.add_argument("--accept-counts", default=None, help="optional comma list of per-dataset caps, formatted DATASET=N")
    materialize.add_argument("--accept-min-margin-chars", type=int, default=DEFAULT_ACCEPT_MIN_MARGIN_CHARS)
    materialize.set_defaults(func=cmd_materialize)

    apply_edits = sub.add_parser("apply-edits", help="create paired original and edited manifests")
    apply_edits.add_argument("--accepted-rewrites", required=True)
    apply_edits.add_argument("--manifest", action="append", help="dataset=manifest.jsonl; may be repeated")
    apply_edits.add_argument("--manifest-map", type=Path, help="JSON object mapping dataset labels to manifest paths")
    apply_edits.add_argument("--original-output", required=True)
    apply_edits.add_argument("--edited-output", required=True)
    apply_edits.set_defaults(func=cmd_apply_edits)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
