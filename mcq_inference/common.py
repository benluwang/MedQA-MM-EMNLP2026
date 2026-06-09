from __future__ import annotations

import io
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field


IMAGE_TOKEN_RE = re.compile(r"<image(?:\s*\d+)?>", re.IGNORECASE)
OPTION_LINE_RE = re.compile(r"(?:^|\n)\s*([A-Z])[\.\):]\s*(.+?)(?=(?:\n\s*[A-Z][\.\):]\s)|$)", re.S)
INLINE_PAREN_OPTION_RE = re.compile(r"(?<![A-Za-z0-9])\(\s*A\s*\)\s*")
INLINE_LETTER_OPTION_RE = re.compile(r"(?<![A-Za-z0-9])A[\.\):]\s+")
CHOICE_PREFIX_RE = re.compile(r"^\s*[A-Z][\.\):]\s*")
ANSWER_CHOICES_BLOCK_RE = re.compile(r"Answer Choices:\s*", re.IGNORECASE)
OPENAI_SAFE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


@dataclass
class MCQExample:
    dataset: str
    example_id: str
    question: str
    options: dict[str, str]
    correct_answer: Optional[str]
    correct_answer_text: Optional[str]
    image_refs: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class MCQPrediction(BaseModel):
    answer: str = Field(description="Single uppercase multiple-choice letter, e.g. A")
    rationale: str = Field(description="Short justification grounded in the question and images.")


def clean_question_text(text: str) -> str:
    text = IMAGE_TOKEN_RE.sub(" ", text or "")
    text = ANSWER_CHOICES_BLOCK_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_answer_choices_block(text: str) -> str:
    text = text or ""
    candidates = [match.start() for match in [
        OPTION_LINE_RE.search(text),
        INLINE_PAREN_OPTION_RE.search(text),
        INLINE_LETTER_OPTION_RE.search(text),
    ] if match]
    if not candidates:
        return clean_question_text(text)
    return clean_question_text(text[: min(candidates)].strip())


def normalize_option_value(text: Any) -> str:
    value = str(text or "").strip()
    value = CHOICE_PREFIX_RE.sub("", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_options(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        out: dict[str, str] = {}
        for key, value in raw.items():
            letter = str(key).strip().upper()
            if not letter:
                continue
            out[letter] = normalize_option_value(value)
        return dict(sorted(out.items()))
    if isinstance(raw, list):
        out: dict[str, str] = {}
        for idx, value in enumerate(raw):
            text = str(value or "").strip()
            match = re.match(r"^\s*([A-Z])[\.\):]\s*(.+)$", text)
            if match:
                out[match.group(1)] = normalize_option_value(match.group(2))
            else:
                out[chr(ord("A") + idx)] = normalize_option_value(text)
        return dict(sorted(out.items()))
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if parsed is None:
            try:
                import ast

                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
        if parsed is not None:
            return normalize_options(parsed)

        matches = OPTION_LINE_RE.findall(text)
        if matches:
            return {
                letter: normalize_option_value(option_text)
                for letter, option_text in matches
            }
    return {}


def answer_letter_from_text(answer: Any) -> Optional[str]:
    text = str(answer or "").strip()
    if not text:
        return None
    match = re.match(r"^\s*([A-Z])(?:[\.\):].*)?$", text)
    if match:
        return match.group(1)
    return text if len(text) == 1 and text.isalpha() else None


def answer_text_from_options(options: dict[str, str], answer: Optional[str]) -> Optional[str]:
    if not answer:
        return None
    return options.get(answer)


def sniff_image_suffix(raw: bytes, fallback: str = ".png") -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return ".webp"
    if raw.startswith(b"BM"):
        return ".bmp"
    return fallback


def extract_answer_from_option_text(options: dict[str, str], answer_text: Any) -> Optional[str]:
    target = normalize_option_value(answer_text)
    if not target:
        return None
    for letter, value in options.items():
        if normalize_option_value(value) == target:
            return letter
    return None


def render_array_to_png(raw: bytes, destination: Path) -> None:
    array = np.load(io.BytesIO(raw))
    image = np.asarray(array).squeeze()
    while image.ndim > 3:
        image = image[0]
    if image.ndim == 3:
        axis = int(np.argmin(image.shape))
        image = np.take(image, image.shape[axis] // 2, axis=axis)
    if image.ndim != 2:
        image = image.reshape(image.shape[-2], image.shape[-1])

    image = np.asarray(image, dtype=np.float32)
    finite_mask = np.isfinite(image)
    if not finite_mask.any():
        image = np.zeros_like(image, dtype=np.uint8)
    else:
        valid = image[finite_mask]
        lo = float(np.percentile(valid, 1))
        hi = float(np.percentile(valid, 99))
        if hi <= lo:
            hi = lo + 1.0
        image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)

    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(destination)


def save_bytes_as_supported_image(raw: bytes, destination_without_suffix: Path, preferred_suffix: str = ".png") -> Path:
    suffix = sniff_image_suffix(raw, fallback=preferred_suffix).lower()
    if suffix in OPENAI_SAFE_SUFFIXES:
        destination = destination_without_suffix.with_suffix(suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        return destination

    destination = destination_without_suffix.with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(raw)) as image:
        image.convert("RGB").save(destination)
    return destination


def write_hf_image_value(value: Any, destination_without_suffix: Path) -> Path:
    if isinstance(value, dict):
        raw = value.get("bytes")
        path = value.get("path")
        if raw is not None:
            if isinstance(raw, memoryview):
                raw = raw.tobytes()
            return save_bytes_as_supported_image(raw, destination_without_suffix, preferred_suffix=".png")
        if path:
            source = Path(path)
            suffix = (source.suffix or ".png").lower()
            if suffix in OPENAI_SAFE_SUFFIXES:
                destination = destination_without_suffix.with_suffix(source.suffix or ".png")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            else:
                destination = destination_without_suffix.with_suffix(".png")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as image:
                    image.convert("RGB").save(destination)
            return destination
        raise ValueError("Unsupported HF image dict: missing both `bytes` and `path`.")

    if isinstance(value, Image.Image):
        destination = destination_without_suffix.with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        value.save(destination)
        return destination

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return save_bytes_as_supported_image(raw, destination_without_suffix, preferred_suffix=".png")

    raise TypeError(f"Unsupported HF image cell type: {type(value)!r}")
