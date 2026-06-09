#!/usr/bin/env python3
"""Core method for auditing device/material shortcut-risk in medical VQA.

The goal of this module is to preserve the main reusable method only:

1. screen question text for generic treatment-device/material cues;
2. render a neutral adjudication prompt for candidate cases;
3. build OpenAI Batch API request lines for human/LLM-assisted review;
4. summarize returned shortcut-risk labels.

It intentionally contains no dataset records, no local paths, no historical
outputs, and no case-specific terms added to recover a preferred result.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
WHITESPACE_RE = re.compile(r"\s+")
LABEL_RE = re.compile(r"\b([A-Z])\b", re.I)


GENERIC_DEVICE_MATERIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "implanted_or_retained_device",
        re.compile(r"\b(?:implant|implanted|prosthesis|prosthetic|hardware|foreign body)\b", re.I),
    ),
    (
        "orthopedic_fixation",
        re.compile(r"\b(?:plate|screw|pin|wire|rod|nail|fixation|cast|splint|brace)\b", re.I),
    ),
    (
        "vascular_or_endovascular_device",
        re.compile(r"\b(?:catheter|stent|coil|filter|graft|balloon|embolization|embolisation)\b", re.I),
    ),
    (
        "tube_line_drain_or_shunt",
        re.compile(r"\b(?:tube|line|drain|shunt|port|lead|electrode)\b", re.I),
    ),
    (
        "closure_or_wound_material",
        re.compile(r"\b(?:suture|staple|dressing|bandage|mesh|cement)\b", re.I),
    ),
    (
        "procedure_or_post_treatment_context",
        re.compile(r"\b(?:postoperative|post-operative|postprocedural|post-procedural|after surgery|treated with|prior surgery|surgical history)\b", re.I),
    ),
)


SYSTEM_PROMPT = """You are auditing medical VQA benchmark cases for shortcut-risk from treatment devices, post-treatment materials, and device/material text cues.

Definitions:
- A device/material cue may be a visible physical device/material, image-embedded text that names a device/material, or question text that names a device/material or a procedure strongly implying one.
- A confirmed shortcut-risk exists when the cue can plausibly identify or strongly favor the correct option without the intended diagnostic or visual reasoning, especially when distractors do not control for the same cue.
- A borderline shortcut-risk exists when the cue is real and meaningfully narrows the answer space, but still requires additional medical reasoning or is only partly decisive.
- A no-shortcut case may still contain a device/material cue if that cue is incidental, nonspecific, part of the intended task evidence, redundant with the question, or insufficient to distinguish the correct option.

Judge the cue, the intended task, and the answer options together. Do not label a case as shortcut-risk merely because a device or procedure is visible or mentioned.

Return concise structured JSON only."""


USER_TEMPLATE = """Audit this case for treatment-device/material shortcut-risk.

Dataset: {dataset}
Example id: {example_id}
Candidate source group: {source_group}
Images sent: {image_count_sent}

Question stem:
{question}

Options:
{options_text}

Correct answer: {correct_answer}. {correct_answer_text}

Text cue screen positives:
{text_screen_summary}

Image feature summary:
{feature_summary}

Task:
1. Confirm which candidate cue sources are real: visible_device, image_text, stem_text.
2. Decide whether the device/material cue is intended evidence, incidental/redundant context, or a possible shortcut.
3. Decide whether weak option design lets the cue identify or strongly favor the correct option.
4. Assign one label: no_shortcut, borderline_shortcut, or confirmed_shortcut.

Return JSON only."""


AUDIT_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "device_material_shortcut_audit",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "confirmed_device_material_cue": {"type": "boolean"},
            "cue_sources_confirmed": {
                "type": "array",
                "items": {"type": "string", "enum": ["visible_device", "image_text", "stem_text"]},
            },
            "cue_terms": {"type": "array", "items": {"type": "string"}},
            "cue_evidence": {"type": "array", "items": {"type": "string"}},
            "cue_is_intended_evidence": {"type": "boolean"},
            "option_design_weakness": {"type": "boolean"},
            "shortcut_risk_label": {
                "type": "string",
                "enum": ["no_shortcut", "borderline_shortcut", "confirmed_shortcut"],
            },
            "shortcut_type": {
                "type": "string",
                "enum": [
                    "visible_device_shortcut",
                    "image_text_device_leakage",
                    "stem_device_context",
                    "intended_or_incidental_device",
                    "mixed_or_other",
                ],
            },
            "rationale": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": [
            "confirmed_device_material_cue",
            "cue_sources_confirmed",
            "cue_terms",
            "cue_evidence",
            "cue_is_intended_evidence",
            "option_design_weakness",
            "shortcut_risk_label",
            "shortcut_type",
            "rationale",
            "confidence",
        ],
    },
}


@dataclass(frozen=True)
class TextCue:
    source: str
    category: str
    term: str
    evidence: str


def clean_space(text: Any) -> str:
    return WHITESPACE_RE.sub(" ", str(text or "").replace("\u00a0", " ")).strip()


def normalize_label(value: Any) -> str | None:
    text = clean_space(value).upper()
    if len(text) == 1 and text in LETTERS:
        return text
    match = LABEL_RE.search(text)
    if match and match.group(1).upper() in LETTERS:
        return match.group(1).upper()
    return None


def normalize_options(raw_options: Any) -> dict[str, str]:
    if isinstance(raw_options, Mapping):
        pairs = list(raw_options.items())
    elif isinstance(raw_options, list):
        pairs = []
        for index, item in enumerate(raw_options):
            default_label = LETTERS[index] if index < len(LETTERS) else str(index + 1)
            if isinstance(item, Mapping):
                label = item.get("label") or item.get("letter") or item.get("key") or default_label
                text = item.get("text") or item.get("option") or item.get("value") or ""
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                label, text = item[0], item[1]
            else:
                label, text = default_label, item
            pairs.append((label, text))
    else:
        return {}

    normalized: dict[str, str] = {}
    for label_raw, text_raw in pairs:
        label = normalize_label(label_raw)
        text = clean_space(text_raw)
        if label and text:
            normalized[label] = text
    return dict(sorted(normalized.items(), key=lambda item: LETTERS.find(item[0]) if item[0] in LETTERS else 999))


def options_text(options: Mapping[str, str]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in options.items())


def evidence_window(text: str, start: int, end: int, radius: int = 80) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return prefix + clean_space(text[left:right]) + suffix


def screen_text_for_device_material_cues(text: str, *, source: str = "stem_text") -> list[TextCue]:
    """High-recall generic screen for device/material cue candidates.

    This function only proposes candidates. It does not decide shortcut-risk.
    """

    cues: list[TextCue] = []
    seen: set[tuple[str, str, str]] = set()
    for category, pattern in GENERIC_DEVICE_MATERIAL_PATTERNS:
        for match in pattern.finditer(text or ""):
            term = clean_space(match.group(0))
            key = (source, category, term.lower())
            if key in seen:
                continue
            seen.add(key)
            cues.append(
                TextCue(
                    source=source,
                    category=category,
                    term=term,
                    evidence=evidence_window(text, match.start(), match.end()),
                )
            )
    return cues


def serialize_text_cues(cues: Iterable[TextCue]) -> str:
    return json.dumps(
        [
            {
                "source": cue.source,
                "category": cue.category,
                "term": cue.term,
                "evidence": cue.evidence,
            }
            for cue in cues
        ],
        ensure_ascii=False,
    )


def compact_json_or_text(value: Any) -> str:
    if value is None or value == "":
        return "[]"
    if isinstance(value, str):
        return clean_space(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def correct_answer_text(record: Mapping[str, Any], options: Mapping[str, str], correct_label: str | None) -> str:
    explicit = clean_space(record.get("correct_answer_text"))
    if explicit:
        return explicit
    if correct_label and correct_label in options:
        return options[correct_label]
    return ""


def source_group_for_record(record: Mapping[str, Any], text_cues: list[TextCue]) -> str:
    parts: list[str] = []
    if record.get("visible_device_candidate"):
        parts.append("visible")
    if record.get("image_text_candidate"):
        parts.append("text:image")
    if text_cues:
        parts.append("text:stem")
    return "+".join(parts) if parts else clean_space(record.get("source_group") or "unspecified")


def render_user_prompt(record: Mapping[str, Any]) -> str:
    options = normalize_options(record.get("options"))
    correct_label = normalize_label(record.get("correct_answer") or record.get("answer"))
    question = clean_space(record.get("question") or record.get("stem"))
    text_sources = record.get("text_sources")
    if isinstance(text_sources, Mapping):
        cue_text = "\n".join(clean_space(value) for value in text_sources.values())
    elif isinstance(text_sources, list):
        cue_text = "\n".join(clean_space(value) for value in text_sources)
    else:
        cue_text = question
    text_cues = screen_text_for_device_material_cues(cue_text)
    image_urls = record.get("image_urls") if isinstance(record.get("image_urls"), list) else []

    return USER_TEMPLATE.format(
        dataset=clean_space(record.get("dataset")),
        example_id=clean_space(record.get("example_id") or record.get("id")),
        source_group=source_group_for_record(record, text_cues),
        image_count_sent=len(image_urls),
        question=question,
        options_text=options_text(options),
        correct_answer=correct_label or clean_space(record.get("correct_answer")),
        correct_answer_text=correct_answer_text(record, options, correct_label),
        text_screen_summary=serialize_text_cues(text_cues),
        feature_summary=compact_json_or_text(record.get("image_feature_summary") or record.get("feature_summary")),
    )


def build_responses_input(system_prompt: str, user_prompt: str, image_urls: Iterable[str] = ()) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
    for image_url in image_urls:
        image_url = clean_space(image_url)
        if image_url:
            user_content.append({"type": "input_image", "image_url": image_url})
    return [
        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


def build_batch_request(
    record: Mapping[str, Any],
    *,
    custom_id: str,
    model: str,
    max_output_tokens: int = 600,
    reasoning_effort: str = "low",
) -> dict[str, Any]:
    image_urls = record.get("image_urls") if isinstance(record.get("image_urls"), list) else []
    body = {
        "model": model,
        "input": build_responses_input(SYSTEM_PROMPT, render_user_prompt(record), image_urls),
        "max_output_tokens": int(max_output_tokens),
        "reasoning": {"effort": reasoning_effort},
        "text": {"format": AUDIT_JSON_SCHEMA},
        "store": False,
    }
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_batch_jsonl(input_path: Path, output_path: Path, *, model: str, prefix: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(iter_jsonl(input_path)):
            request = build_batch_request(record, custom_id=f"{prefix}-{index:06d}", model=model)
            handle.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")


def extract_response_text(body: Mapping[str, Any]) -> str:
    texts: list[str] = []
    for item in body.get("output") or []:
        for content in item.get("content") or []:
            if isinstance(content, Mapping) and content.get("text"):
                texts.append(str(content["text"]))
    if texts:
        return "\n".join(texts)
    return str(body.get("output_text") or "")


def parse_audit_payload(text: str) -> dict[str, Any] | None:
    text = clean_space(text)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def summarize_batch_output(batch_output_path: Path, csv_path: Path) -> None:
    """Write aggregate label counts from Batch API output JSONL."""

    counts: dict[str, int] = {}
    for row in iter_jsonl(batch_output_path):
        body = ((row.get("response") or {}).get("body") or {})
        payload = parse_audit_payload(extract_response_text(body)) or {}
        label = payload.get("shortcut_risk_label") or "parse_error"
        counts[label] = counts.get(label, 0) + 1

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["shortcut_risk_label", "n"])
        writer.writeheader()
        for label, count in sorted(counts.items()):
            writer.writerow({"shortcut_risk_label": label, "n": count})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    make_batch = sub.add_parser("make-batch", help="Render Batch API JSONL from generic candidate records.")
    make_batch.add_argument("--input", type=Path, required=True)
    make_batch.add_argument("--output", type=Path, required=True)
    make_batch.add_argument("--model", default="gpt-5.4")
    make_batch.add_argument("--prefix", default="device-material-shortcut")

    summarize = sub.add_parser("summarize", help="Summarize shortcut labels from Batch API output JSONL.")
    summarize.add_argument("--batch-output", type=Path, required=True)
    summarize.add_argument("--csv", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "make-batch":
        write_batch_jsonl(args.input, args.output, model=args.model, prefix=args.prefix)
    elif args.command == "summarize":
        summarize_batch_output(args.batch_output, args.csv)


if __name__ == "__main__":
    main()
