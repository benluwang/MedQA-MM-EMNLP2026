#!/usr/bin/env python3
"""Core logic for visible image-text shortcut auditing and repair.

This module intentionally contains only reusable method logic and prompts.
It does not contain dataset examples, result files, absolute paths, API keys,
or project-specific output directories.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


FINAL_CLASSES = (
    "no_issue",
    "balanced_option_visibility",
    "legitimate_image_text_evidence",
    "easy_actionable_image_text_shortcut",
    "hard_actionable_image_text_shortcut",
    "not_image_editable",
)

REPAIR_STRATEGIES = (
    "minimal_redaction",
    "full_nonessential_text_removal",
    "neutral_relabeling",
    "crop_or_layout_repair",
    "evidence_preserving_diagram_simplification",
)

ANSWER_SPECIFICITY = ("correct_only", "correct_dominant", "balanced", "weak_or_noise")
REPAIR_ACTIONS = ("crop", "mask", "inpaint", "neutral_relabel", "drop_image", "keep_unchanged", "exclude_from_repair")


ADJUDICATION_SYSTEM_PROMPT = """You audit medical multiple-choice VQA cases for visible image-text shortcut leakage.

Scope rules:
- Only consider text visibly rendered inside the evaluation-visible images.
- Do NOT consider leakage from question text, clinical history, imaging findings fields, metadata captions, diagnosis fields, explanations, or answer rationales.
- Clinical history, imaging findings, and answer options are normal model inputs used only to understand the intended task.
- Captions or labels count only if they are visible as pixels inside the image.
- Do not use OCR supplied by a tool; inspect the images yourself.

Definitions:
- no_issue: no visible image text overlap with the correct answer, or any apparent overlap is too ambiguous.
- balanced_option_visibility: the correct answer and multiple distractors are similarly visible, so image text does not uniquely cue the correct answer.
- legitimate_image_text_evidence: visible text is intended task evidence, e.g. lab tables, axes, legends, statistical tables, chemical formulas, food exposure tables, or row/column labels required to answer.
- easy_actionable_image_text_shortcut: a small, localized, nonessential visible text artifact such as a diagnosis label, panel title, caption, overlay, filename/source label, or annotation directly names or strongly cues the correct answer and can be removed while preserving the same medical image.
- hard_actionable_image_text_shortcut: visible text is a true shortcut but intertwined with anatomical/structural/diagram semantics; repair needs careful candidate generation. Prefer this class when only a subset of images or labels are problematic and a repair could plausibly neutralize labels while preserving useful visual evidence.
- not_image_editable: visible image text is task-essential or the whole central image is a text-heavy explanatory/teaching answer figure, table, comparison chart, or answer-explanation panel. Classify these as not_image_editable even if the recommended action is to drop that image. Reserve easy_actionable for small local text artifacts, not full text tables or explanatory figures.

Return JSON only."""


ADJUDICATION_JSON_SCHEMA = {
    "case_id": "string",
    "dataset": "string",
    "final_class": "no_issue | balanced_option_visibility | legitimate_image_text_evidence | easy_actionable_image_text_shortcut | hard_actionable_image_text_shortcut | not_image_editable",
    "answer_specificity": "correct_only | correct_dominant | balanced | weak_or_noise",
    "problematic_image_ids": ["string"],
    "visible_text_evidence": [
        {
            "image_id": "string",
            "visible_text": "string",
            "why_it_matters": "string",
        }
    ],
    "image_text_function": "diagnosis_label | caption | panel_title | dicom_overlay | annotation_label | table_text | axis_label | legend | row_column_label | other",
    "repair_strategy_hint": "string",
    "recommended_action": "crop | mask | inpaint | neutral_relabel | drop_image | keep_unchanged | exclude_from_repair",
    "risk_to_item_validity": "low | medium | high",
    "image2_prompt": "string",
    "rationale": "string",
}


DIRECT_REPAIR_PROMPT_TEMPLATE = """Edit only the visible rendered text artifact that reveals or strongly cues the answer option.

Remove or neutralize the following visible text evidence:
{visible_text_evidence}

Target image id: {image_id}
Recommended action: {recommended_action}

Rules:
- Do not alter medical image content, anatomy, lesions, devices, measurements, arrows, calipers, scale bars, modality appearance, or clinically relevant visual patterns.
- If the text is at a border, title, footer, or caption area, crop or mask that area cleanly while preserving the central medical image.
- If an annotation label must remain for orientation, replace answer-revealing words with neutral placeholders such as "Region A", "Structure 1", "Panel A", or "View 1".
- Do not add diagnosis names, disease labels, answer option text, explanatory notes, or new visual findings.
- Keep the rest of the image unchanged."""


HARD_REPAIR_STRATEGY_PROMPTS = {
    "minimal_redaction": """Remove only the exact answer-revealing visible text. Preserve all other labels, arrows, leader lines, panel boundaries, and medical image content.""",
    "full_nonessential_text_removal": """Remove all nonessential visible text, captions, panel titles, source labels, and overlays that could cue an answer. Preserve medical image content and task-essential visual evidence.""",
    "neutral_relabeling": """Replace answer-revealing labels with neutral placeholders such as "Region A", "Region B", "Structure 1", "Structure 2", "Panel A", and "Panel B". Keep arrows, leader lines, circles, and layout unchanged.""",
    "crop_or_layout_repair": """Crop out or cleanly remove border text, title text, footer text, DICOM overlay, or embedded caption while preserving the central medical image and clinically relevant visual patterns.""",
    "evidence_preserving_diagram_simplification": """For schematic or educational diagrams, preserve the visual structure but remove answer-specific labels. Use generic labels only. Do not add new medical findings or explanatory text.""",
}


TEXT_SHORTCUT_JUDGE_PROMPT = """You are a strict model-only judge for repaired medical benchmark images.
Do not use OCR tools. Inspect the original and edited images yourself.
Judge only visible text artifacts and answer-option shortcut risk. Return JSON only.

Judge gates:
1. leak_removed: the original answer-revealing visible text is no longer human-visible in the edited image.
2. no_new_shortcut: the edited image does not introduce new text, labels, symbols, or artifacts that cue the correct answer.

Return:
{
  "candidate_id": "...",
  "leak_removed": true,
  "no_new_shortcut": true,
  "original_shortcut_still_visible": false,
  "new_shortcut_description": "",
  "rationale": "..."
}"""


FIDELITY_ITEM_JUDGE_PROMPT = """You are a strict model-only judge for medical image repair.
Do not use OCR tools. Compare the original and edited images yourself.
Judge whether medical visual content and item validity were preserved. Return JSON only.

Judge gates:
1. medical_fidelity_preserved: the edited image preserves lesions, anatomy, devices, measurements, arrows/calipers, modality appearance, and clinically relevant visual patterns.
2. item_still_answerable: using the allowed clinical input, options, and edited image, the item remains a valid multiple-choice VQA item.
3. original_correct_answer_still_reasonable: the original correct answer remains reasonable after the edit.

Return:
{
  "candidate_id": "...",
  "medical_fidelity_preserved": true,
  "item_still_answerable": true,
  "original_correct_answer_still_reasonable": true,
  "changed_medical_content": "",
  "rationale": "..."
}"""


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    image_url: str
    display_name: str | None = None


@dataclass(frozen=True)
class MCQCase:
    dataset: str
    case_id: str
    question: str
    options: dict[str, str]
    correct_answer: str
    correct_answer_text: str
    images: tuple[ImageRef, ...]


@dataclass(frozen=True)
class RepairCandidateSpec:
    candidate_id: str
    dataset: str
    case_id: str
    image_id: str
    strategy: str
    sample_index: int
    prompt: str
    source_file_id: str | None = None


def image_file_to_data_url(path: str | Path) -> str:
    """Encode an image as a data URL for Responses API examples or local tests."""
    image_path = Path(path)
    mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))


def case_context(case: MCQCase) -> str:
    images = "\n".join(
        f"- {image.image_id}: {image.display_name or image.image_id}"
        for image in case.images
    )
    return (
        f"Dataset: {case.dataset}\n"
        f"Case ID: {case.case_id}\n\n"
        f"Question / clinical input:\n{case.question}\n\n"
        f"Options:\n{format_options(case.options)}\n\n"
        f"Correct answer: {case.correct_answer} - {case.correct_answer_text}\n\n"
        f"Evaluation-visible images:\n{images}"
    )


def build_adjudication_request(
    case: MCQCase,
    *,
    model: str,
    custom_id: str,
    max_output_tokens: int = 1800,
    reasoning_effort: str = "medium",
) -> dict[str, Any]:
    """Build one Batch-compatible /v1/responses request for image-text adjudication."""
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                f"{case_context(case)}\n\n"
                "Task: classify whether visible text pixels in the images create an answer-option shortcut. "
                "Ignore all non-visible metadata as leakage evidence. If repairable, write a concise edit prompt "
                "that targets only rendered text artifacts and explicitly preserves medical visual findings.\n\n"
                f"Expected JSON shape:\n{json.dumps(ADJUDICATION_JSON_SCHEMA, indent=2)}"
            ),
        }
    ]
    for image in case.images:
        content.append({"type": "input_text", "text": f"Image ID: {image.image_id}"})
        content.append({"type": "input_image", "image_url": image.image_url, "detail": "high"})
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [
                {"role": "system", "content": ADJUDICATION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": reasoning_effort},
        },
    }


def visible_text_summary(adjudication: dict[str, Any]) -> str:
    rows = adjudication.get("visible_text_evidence") or []
    if not rows:
        return "(none provided)"
    return "\n".join(
        f"- {row.get('image_id', '')}: {row.get('visible_text', '')} -- {row.get('why_it_matters', '')}"
        for row in rows
    )


def direct_repair_prompt(adjudication: dict[str, Any], *, image_id: str) -> str:
    return DIRECT_REPAIR_PROMPT_TEMPLATE.format(
        visible_text_evidence=visible_text_summary(adjudication),
        image_id=image_id,
        recommended_action=adjudication.get("recommended_action", "inpaint"),
    )


def repair_candidate_specs(
    *,
    case: MCQCase,
    adjudication: dict[str, Any],
    image_file_ids: dict[str, str] | None = None,
    hard_samples_per_strategy: int = 2,
) -> list[RepairCandidateSpec]:
    """Create easy 1-shot or hard 5x2 repair candidates from adjudication output."""
    image_file_ids = image_file_ids or {}
    final_class = adjudication.get("final_class")
    image_ids = adjudication.get("problematic_image_ids") or []
    specs: list[RepairCandidateSpec] = []
    if final_class == "easy_actionable_image_text_shortcut":
        for image_id in image_ids:
            specs.append(
                RepairCandidateSpec(
                    candidate_id=f"{case.dataset}::{case.case_id}::{image_id}::direct_edit::1",
                    dataset=case.dataset,
                    case_id=case.case_id,
                    image_id=image_id,
                    strategy="direct_edit",
                    sample_index=1,
                    prompt=adjudication.get("image2_prompt") or direct_repair_prompt(adjudication, image_id=image_id),
                    source_file_id=image_file_ids.get(image_id),
                )
            )
    elif final_class == "hard_actionable_image_text_shortcut":
        for image_id in image_ids:
            for strategy in REPAIR_STRATEGIES:
                for sample_index in range(1, hard_samples_per_strategy + 1):
                    prompt = (
                        f"{direct_repair_prompt(adjudication, image_id=image_id)}\n\n"
                        f"Strategy: {strategy}\n"
                        f"{HARD_REPAIR_STRATEGY_PROMPTS[strategy]}"
                    )
                    specs.append(
                        RepairCandidateSpec(
                            candidate_id=f"{case.dataset}::{case.case_id}::{image_id}::{strategy}::{sample_index}",
                            dataset=case.dataset,
                            case_id=case.case_id,
                            image_id=image_id,
                            strategy=strategy,
                            sample_index=sample_index,
                            prompt=prompt,
                            source_file_id=image_file_ids.get(image_id),
                        )
                    )
    return specs


def build_image_edit_request(
    candidate: RepairCandidateSpec,
    *,
    model: str = "gpt-image-2",
    quality: str = "low",
    size: str = "auto",
    output_format: str = "png",
) -> dict[str, Any]:
    """Build one Batch-compatible /v1/images/edits request."""
    if not candidate.source_file_id:
        raise ValueError("source_file_id is required for image edit batch requests")
    return {
        "custom_id": candidate.candidate_id,
        "method": "POST",
        "url": "/v1/images/edits",
        "body": {
            "model": model,
            "images": [{"file_id": candidate.source_file_id}],
            "prompt": candidate.prompt,
            "quality": quality,
            "size": size,
            "output_format": output_format,
        },
    }


def build_text_shortcut_judge_request(
    *,
    case: MCQCase,
    candidate_id: str,
    original_image_url: str,
    edited_image_url: str,
    model: str,
    custom_id: str,
    max_output_tokens: int = 1200,
) -> dict[str, Any]:
    prompt = f"{case_context(case)}\n\nCandidate ID: {candidate_id}\n\n{TEXT_SHORTCUT_JUDGE_PROMPT}"
    return _paired_image_judge_request(
        custom_id=custom_id,
        model=model,
        prompt=prompt,
        original_image_url=original_image_url,
        edited_image_url=edited_image_url,
        max_output_tokens=max_output_tokens,
    )


def build_fidelity_item_judge_request(
    *,
    case: MCQCase,
    candidate_id: str,
    original_image_url: str,
    edited_image_url: str,
    model: str,
    custom_id: str,
    max_output_tokens: int = 1200,
) -> dict[str, Any]:
    prompt = f"{case_context(case)}\n\nCandidate ID: {candidate_id}\n\n{FIDELITY_ITEM_JUDGE_PROMPT}"
    return _paired_image_judge_request(
        custom_id=custom_id,
        model=model,
        prompt=prompt,
        original_image_url=original_image_url,
        edited_image_url=edited_image_url,
        max_output_tokens=max_output_tokens,
    )


def _paired_image_judge_request(
    *,
    custom_id: str,
    model: str,
    prompt: str,
    original_image_url: str,
    edited_image_url: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_text", "text": "Original image:"},
                        {"type": "input_image", "image_url": original_image_url, "detail": "high"},
                        {"type": "input_text", "text": "Edited image:"},
                        {"type": "input_image", "image_url": edited_image_url, "detail": "high"},
                    ],
                }
            ],
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": "medium"},
        },
    }


def dataset_ready_from_judges(
    *,
    has_edited_image: bool,
    text_judge: dict[str, Any],
    fidelity_judge: dict[str, Any],
) -> tuple[bool, str]:
    """Return (dataset_ready, primary_failure_gate)."""
    if not has_edited_image:
        return False, "missing_or_batch_error"
    if not bool(text_judge.get("leak_removed")):
        return False, "leak_not_removed"
    if not bool(text_judge.get("no_new_shortcut")):
        return False, "new_shortcut"
    if not bool(fidelity_judge.get("medical_fidelity_preserved")):
        return False, "fidelity_changed"
    if not bool(fidelity_judge.get("original_correct_answer_still_reasonable")):
        return False, "fidelity_changed"
    if not bool(fidelity_judge.get("item_still_answerable")):
        return False, "not_answerable"
    return True, "none"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Closed-form pass@k for n generated candidates and c usable candidates."""
    if n <= 0 or k <= 0:
        return 0.0
    k = min(k, n)
    c = max(0, min(c, n))
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def summarize_pass_at_k(
    rows: Iterable[dict[str, Any]],
    *,
    case_fields: tuple[str, str] = ("dataset", "case_id"),
    ready_field: str = "dataset_ready",
    max_k: int = 10,
) -> dict[str, float | None]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row[case_fields[0]]), str(row[case_fields[1]])), []).append(row)
    out: dict[str, float | None] = {}
    for k in range(1, max_k + 1):
        values = [
            pass_at_k(len(case_rows), sum(bool(row.get(ready_field)) for row in case_rows), k)
            for case_rows in grouped.values()
        ]
        out[f"pass@{k}"] = (sum(values) / len(values)) if values else None
    return out


def answer_transition(original_correct: bool, repaired_correct: bool) -> Literal[
    "correct_to_correct",
    "correct_to_wrong",
    "wrong_to_correct",
    "wrong_to_wrong",
]:
    if original_correct and repaired_correct:
        return "correct_to_correct"
    if original_correct and not repaired_correct:
        return "correct_to_wrong"
    if (not original_correct) and repaired_correct:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def paired_before_after_summary(pairs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize paired VQA predictions.

    Each row should include bool fields `original_correct` and `repaired_correct`.
    """
    counts = {
        "n": 0,
        "original_correct": 0,
        "repaired_correct": 0,
        "correct_to_correct": 0,
        "correct_to_wrong": 0,
        "wrong_to_correct": 0,
        "wrong_to_wrong": 0,
    }
    for row in pairs:
        original = bool(row.get("original_correct"))
        repaired = bool(row.get("repaired_correct"))
        counts["n"] += 1
        counts["original_correct"] += int(original)
        counts["repaired_correct"] += int(repaired)
        counts[answer_transition(original, repaired)] += 1
    n = counts["n"]
    return {
        **counts,
        "original_accuracy": (counts["original_correct"] / n) if n else None,
        "repaired_accuracy": (counts["repaired_correct"] / n) if n else None,
        "delta_accuracy": ((counts["repaired_correct"] - counts["original_correct"]) / n) if n else None,
    }


def parse_response_json(response_body: dict[str, Any]) -> Any:
    """Extract JSON from a Responses API body."""
    chunks: list[str] = []
    for item in response_body.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "input_text"}:
                chunks.append(str(content.get("text") or ""))
    text = "".join(chunks).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}

