#!/usr/bin/env python3
"""Validate or locally reconstruct the source-linked MEDQA-MM subset.

The public MEDQA-MM release is identifier-only. This script resolves the
released locators against three official source repositories and can, at the
user's explicit request, materialize a local standardized copy for evaluation.

It never uploads or republishes source payloads. In particular, the
MedXpertQA paper asks users not to share benchmark examples online and limits
the data to model or human medical evaluation. Read and follow every source's
terms before using ``--materialize``.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import hf_hub_download
from PIL import Image
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IDS_PATH = SCRIPT_DIR / "medqa_mm_ids.json"
DEFAULT_OUTPUT_DIR = Path.cwd() / "medqa_mm_reconstructed"

SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "MedThinkVQA": {
        "repo_id": "bio-nlp-umass/MedThinkVQA",
        "revision": "5555b80e167c4796efab92fe709fb05914077dc0",
        "config": "default",
        "split": "test",
        "metadata_file": "test.jsonl",
        "license": "CC BY-NC-SA 4.0",
        "terms_url": "https://huggingface.co/datasets/bio-nlp-umass/MedThinkVQA",
    },
    "MedXpertQA": {
        "repo_id": "TsinghuaC3I/MedXpertQA",
        "revision": "7e7c465a68eb2b866926bfa59c8c9d17a8daba65",
        "config": "MM",
        "split": "test",
        "metadata_file": "MM/test.jsonl",
        "images_file": "images.zip",
        "license": "See source terms; evaluation-only/no-online-sharing statement in paper",
        "terms_url": "https://proceedings.mlr.press/v267/zuo25a.html",
    },
    "MMMU-HM": {
        "repo_id": "MMMU/MMMU",
        "revision": "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68",
        "split": "test",
        "license": "Apache-2.0; source-level copyright disclaimer also applies",
        "terms_url": "https://huggingface.co/datasets/MMMU/MMMU",
        "subsets": [
            "Basic_Medical_Science",
            "Clinical_Medicine",
            "Diagnostics_and_Laboratory_Medicine",
            "Pharmacy",
            "Public_Health",
        ],
    },
}

EXPECTED_COUNTS = {
    "MedThinkVQA": 207,
    "MedXpertQA": 450,
    "MMMU-HM": 343,
}

MEDTHINK_QUESTION = "What is the most likely diagnosis?"
IMAGE_TOKEN_RE = re.compile(r"<image(?:\s*\d+)?>", re.IGNORECASE)
CHOICE_PREFIX_RE = re.compile(r"^\s*[A-Z][\.\):]\s*")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class ReconstructionError(RuntimeError):
    """Raised when released locators cannot be resolved exactly."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate MEDQA-MM identifiers or materialize a local evaluation copy "
            "from the official source datasets."
        )
    )
    parser.add_argument(
        "--ids",
        type=Path,
        default=DEFAULT_IDS_PATH,
        help="Path to medqa_mm_ids.json or a JSONL locator file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="New directory for validation outputs and optional local payload.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face download cache directory.",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Also create local standardized examples and image files.",
    )
    parser.add_argument(
        "--accept-source-terms",
        action="store_true",
        help="Confirm that you reviewed and accept all upstream source terms.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReconstructionError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def load_locators(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ReconstructionError(f"Identifier file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        records = load_jsonl(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            records = payload.get("records")
        else:
            records = payload
        if not isinstance(records, list):
            raise ReconstructionError(f"No record list found in {path}")
    if len(records) != 1000:
        raise ReconstructionError(f"Expected 1,000 locators, found {len(records)}")
    required = {"medqa_mm_id", "dataset", "split", "example_id", "row_index"}
    for index, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            raise ReconstructionError(f"Locator {index} is missing {sorted(missing)}")
    medqa_ids = [str(record["medqa_mm_id"]) for record in records]
    if len(set(medqa_ids)) != len(medqa_ids):
        raise ReconstructionError("Duplicate medqa_mm_id values detected")
    counts = Counter(str(record["dataset"]) for record in records)
    if dict(counts) != EXPECTED_COUNTS:
        raise ReconstructionError(f"Unexpected source counts: {dict(counts)}")
    return records


def download_file(
    source_name: str,
    filename: str,
    cache_dir: Path | None,
) -> Path:
    spec = SOURCE_SPECS[source_name]
    return Path(
        hf_hub_download(
            repo_id=spec["repo_id"],
            filename=filename,
            repo_type="dataset",
            revision=spec["revision"],
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )


def load_source_rows(
    materialize: bool,
    cache_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    medthink_path = download_file(
        "MedThinkVQA",
        SOURCE_SPECS["MedThinkVQA"]["metadata_file"],
        cache_dir,
    )
    medthink_rows = load_jsonl(medthink_path)

    medxpert_path = download_file(
        "MedXpertQA",
        SOURCE_SPECS["MedXpertQA"]["metadata_file"],
        cache_dir,
    )
    medxpert_rows = load_jsonl(medxpert_path)

    mmmu_rows: dict[str, list[dict[str, Any]]] = {}
    for subset in SOURCE_SPECS["MMMU-HM"]["subsets"]:
        parquet_path = download_file(
            "MMMU-HM",
            f"{subset}/test-00000-of-00001.parquet",
            cache_dir,
        )
        columns = None if materialize else ["id", "question_type"]
        mmmu_rows[subset] = pq.read_table(parquet_path, columns=columns).to_pylist()

    source_counts = {
        "MedThinkVQA": len(medthink_rows),
        "MedXpertQA": len(medxpert_rows),
        "MMMU-HM": sum(
            1
            for rows in mmmu_rows.values()
            for row in rows
            if str(row.get("question_type")) == "multiple-choice"
        ),
    }
    expected_source_counts = {
        "MedThinkVQA": 720,
        "MedXpertQA": 2000,
        "MMMU-HM": 1712,
    }
    if source_counts != expected_source_counts:
        raise ReconstructionError(
            f"Source counts changed: expected {expected_source_counts}, found {source_counts}"
        )
    return {
        "MedThinkVQA": medthink_rows,
        "MedXpertQA": medxpert_rows,
        "MMMU-HM": mmmu_rows,
    }, source_counts


def source_record_for_locator(
    locator: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    dataset = str(locator["dataset"])
    row_index = int(locator["row_index"])
    example_id = str(locator["example_id"])

    if dataset == "MedThinkVQA":
        rows = sources[dataset]
        if not 0 <= row_index < len(rows):
            raise ReconstructionError(f"MedThinkVQA row_index out of range: {row_index}")
        row = rows[row_index]
        resolved_id = f"test_{row_index}_{row.get('title', 'case')}"
        if resolved_id != example_id:
            raise ReconstructionError(
                f"MedThinkVQA locator mismatch: {example_id!r} != {resolved_id!r}"
            )
        case_number = locator.get("case_number")
        if case_number is not None and str(case_number) not in str(row.get("title", "")):
            raise ReconstructionError(
                f"MedThinkVQA case number mismatch for {locator['medqa_mm_id']}"
            )
        return row

    if dataset == "MedXpertQA":
        rows = sources[dataset]
        if not 0 <= row_index < len(rows):
            raise ReconstructionError(f"MedXpertQA row_index out of range: {row_index}")
        row = rows[row_index]
        if str(row.get("id")) != example_id:
            by_id = {str(candidate.get("id")): candidate for candidate in rows}
            if example_id not in by_id:
                raise ReconstructionError(f"MedXpertQA ID not found: {example_id}")
            row = by_id[example_id]
        return row

    if dataset == "MMMU-HM":
        subset = str(locator.get("mmmu_subset") or "")
        if subset not in sources[dataset]:
            raise ReconstructionError(f"Unknown MMMU subset: {subset}")
        rows = sources[dataset][subset]
        if not 0 <= row_index < len(rows):
            raise ReconstructionError(f"MMMU row_index out of range: {subset}/{row_index}")
        row = rows[row_index]
        if str(row.get("id")) != example_id:
            by_id = {str(candidate.get("id")): candidate for candidate in rows}
            if example_id not in by_id:
                raise ReconstructionError(f"MMMU ID not found: {example_id}")
            row = by_id[example_id]
        if str(row.get("question_type")) != "multiple-choice":
            raise ReconstructionError(f"Selected MMMU item is not multiple-choice: {example_id}")
        return row

    raise ReconstructionError(f"Unsupported source dataset: {dataset}")


def resolved_locator(locator: dict[str, Any]) -> dict[str, Any]:
    dataset = str(locator["dataset"])
    spec = SOURCE_SPECS[dataset]
    record = {
        "medqa_mm_id": str(locator["medqa_mm_id"]),
        "dataset": dataset,
        "source_dataset": str(locator["source_dataset"]),
        "source_repo": spec["repo_id"],
        "source_revision": spec["revision"],
        "source_config": (
            str(locator.get("mmmu_subset"))
            if dataset == "MMMU-HM"
            else spec.get("config")
        ),
        "split": str(locator["split"]),
        "example_id": str(locator["example_id"]),
        "row_index": int(locator["row_index"]),
        "source_license_or_terms": spec["license"],
        "source_terms_url": spec["terms_url"],
    }
    if locator.get("case_number") is not None:
        record["case_number"] = int(locator["case_number"])
    if locator.get("mmmu_subset"):
        record["mmmu_subset"] = str(locator["mmmu_subset"])
    return record


def clean_text(value: Any) -> str:
    text = IMAGE_TOKEN_RE.sub(" ", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def strip_answer_choices(value: Any) -> str:
    text = str(value or "")
    marker = re.search(r"Answer Choices\s*:", text, flags=re.IGNORECASE)
    if marker:
        text = text[: marker.start()]
    return clean_text(text)


def normalize_options(raw: Any) -> dict[str, str]:
    if isinstance(raw, str):
        stripped = raw.strip()
        try:
            raw = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            raw = [part.strip() for part in stripped.split("|") if part.strip()]
    if isinstance(raw, dict):
        values = [(str(key).strip().upper(), value) for key, value in raw.items()]
    elif isinstance(raw, (list, tuple)):
        values = [(chr(ord("A") + index), value) for index, value in enumerate(raw)]
    else:
        raise ReconstructionError(f"Unsupported options value: {type(raw).__name__}")
    options: dict[str, str] = {}
    for letter, value in values:
        normalized = CHOICE_PREFIX_RE.sub("", str(value or "").strip())
        options[letter] = re.sub(r"\s+", " ", normalized).strip()
    return dict(sorted(options.items()))


def normalize_answer(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    match = re.search(r"\b([A-Z])\b", text)
    return match.group(1) if match else None


def safe_filename(value: str, fallback: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", value).strip("._")
    return cleaned[:180] or fallback


def copy_medthink_images(
    row: dict[str, Any],
    destination: Path,
    cache_dir: Path | None,
) -> list[str]:
    paths: list[str] = []
    image_count = int(row.get("image_count") or 0)
    for image_index in range(1, image_count + 1):
        source_path = row.get(f"image_{image_index:02d}_path")
        if not source_path:
            continue
        cached = download_file("MedThinkVQA", str(source_path), cache_dir)
        filename = safe_filename(Path(str(source_path)).name, f"image_{image_index:02d}.png")
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, target)
        paths.append(target.name)
    if not paths:
        raise ReconstructionError("Selected MedThinkVQA item has no materialized images")
    return paths


def build_zip_member_map(archive: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    duplicates: set[str] = set()
    for member in archive.namelist():
        basename = Path(member).name
        if not basename:
            continue
        if basename in members and members[basename] != member:
            duplicates.add(basename)
        else:
            members[basename] = member
    for basename in duplicates:
        members.pop(basename, None)
    return members


def extract_medxpert_images(
    row: dict[str, Any],
    destination: Path,
    archive: zipfile.ZipFile,
    members: dict[str, str],
) -> list[str]:
    paths: list[str] = []
    for index, raw_name in enumerate(row.get("images") or [], start=1):
        basename = Path(str(raw_name)).name
        member = members.get(basename)
        if not member:
            raise ReconstructionError(f"MedXpertQA image missing from archive: {raw_name}")
        target = destination / safe_filename(basename, f"image_{index:02d}.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(member))
        paths.append(target.name)
    if not paths:
        raise ReconstructionError("Selected MedXpertQA item has no materialized images")
    return paths


def save_arrow_image(value: Any, target_base: Path) -> Path | None:
    if value is None:
        return None
    target_base.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        raw = value.get("bytes")
        source_path = value.get("path")
        if raw:
            with Image.open(io.BytesIO(raw)) as image:
                target = target_base.with_suffix(".png")
                image.save(target, format="PNG")
                return target
        if source_path and Path(str(source_path)).is_file():
            suffix = Path(str(source_path)).suffix or ".png"
            target = target_base.with_suffix(suffix)
            shutil.copy2(Path(str(source_path)), target)
            return target
    if isinstance(value, (bytes, bytearray)):
        with Image.open(io.BytesIO(value)) as image:
            target = target_base.with_suffix(".png")
            image.save(target, format="PNG")
            return target
    if isinstance(value, Image.Image):
        target = target_base.with_suffix(".png")
        value.save(target, format="PNG")
        return target
    raise ReconstructionError(f"Unsupported MMMU image value: {type(value).__name__}")


def materialize_mmmu_images(row: dict[str, Any], destination: Path) -> list[str]:
    paths: list[str] = []
    for image_index in range(1, 8):
        target = save_arrow_image(
            row.get(f"image_{image_index}"),
            destination / f"image_{image_index:02d}",
        )
        if target:
            paths.append(target.name)
    if not paths:
        raise ReconstructionError("Selected MMMU item has no materialized images")
    return paths


def relative_image_paths(
    output_dir: Path,
    item_dir: Path,
    names: Iterable[str],
) -> list[str]:
    return [str((item_dir / name).relative_to(output_dir)).replace("\\", "/") for name in names]


def standardize_example(
    locator: dict[str, Any],
    row: dict[str, Any],
    output_dir: Path,
    cache_dir: Path | None,
    medxpert_archive: zipfile.ZipFile | None,
    medxpert_members: dict[str, str] | None,
) -> dict[str, Any]:
    dataset = str(locator["dataset"])
    medqa_id = str(locator["medqa_mm_id"])
    item_dir = output_dir / "images" / medqa_id

    if dataset == "MedThinkVQA":
        clinical_history = clean_text(row.get("CLINICAL_HISTORY"))
        question = f"{MEDTHINK_QUESTION}\n\nClinical history: {clinical_history}"
        options = normalize_options(row.get("options"))
        answer = normalize_answer(row.get("correct_answer"))
        names = copy_medthink_images(row, item_dir, cache_dir)
        metadata = {
            "title": row.get("title"),
            "image_count": int(row.get("image_count") or 0),
        }
    elif dataset == "MedXpertQA":
        if medxpert_archive is None or medxpert_members is None:
            raise ReconstructionError("MedXpertQA image archive was not initialized")
        question = strip_answer_choices(row.get("question"))
        options = normalize_options(row.get("options"))
        answer = normalize_answer(row.get("label"))
        names = extract_medxpert_images(
            row,
            item_dir,
            medxpert_archive,
            medxpert_members,
        )
        metadata = {
            "medical_task": row.get("medical_task"),
            "body_system": row.get("body_system"),
            "question_type": row.get("question_type"),
        }
    elif dataset == "MMMU-HM":
        question = clean_text(row.get("question"))
        options = normalize_options(row.get("options"))
        answer = normalize_answer(row.get("answer"))
        names = materialize_mmmu_images(row, item_dir)
        metadata = {
            "mmmu_subset": locator.get("mmmu_subset"),
            "question_type": row.get("question_type"),
            "subfield": row.get("subfield"),
            "topic_difficulty": row.get("topic_difficulty"),
            "img_type": row.get("img_type"),
        }
    else:
        raise ReconstructionError(f"Unsupported source dataset: {dataset}")

    correct_answer_text = options.get(answer) if answer else None
    source = resolved_locator(locator)
    return {
        "medqa_mm_id": medqa_id,
        "dataset": dataset,
        "source_example_id": str(locator["example_id"]),
        "question": question,
        "options": options,
        "correct_answer": answer,
        "correct_answer_text": correct_answer_text,
        "image_paths": relative_image_paths(output_dir, item_dir, names),
        "source": source,
        "metadata": metadata,
        "release_variant": "source-linked reconstruction",
        "redistribution_notice": (
            "Local evaluation copy only. Source licenses and terms apply; do not redistribute."
        ),
    }


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReconstructionError(
            f"Output directory is not empty: {output_dir}. Choose a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    args = parse_args()
    if args.materialize and not args.accept_source_terms:
        print(
            "ERROR: --materialize requires --accept-source-terms after reviewing all upstream terms.",
            file=sys.stderr,
        )
        return 2

    try:
        locators = load_locators(args.ids.resolve())
        prepare_output_dir(args.output_dir.resolve())
        output_dir = args.output_dir.resolve()
        cache_dir = args.cache_dir.resolve() if args.cache_dir else None

        sources, source_counts = load_source_rows(args.materialize, cache_dir)
        matched: dict[str, dict[str, Any]] = {}
        resolved_rows: list[dict[str, Any]] = []
        for locator in locators:
            medqa_id = str(locator["medqa_mm_id"])
            matched[medqa_id] = source_record_for_locator(locator, sources)
            resolved_rows.append(resolved_locator(locator))

        write_jsonl(output_dir / "resolved_ids.jsonl", resolved_rows)

        summary: dict[str, Any] = {
            "status": "validated",
            "record_count": len(locators),
            "dataset_counts": dict(Counter(row["dataset"] for row in locators)),
            "source_counts": source_counts,
            "source_revisions": {
                name: spec["revision"] for name, spec in SOURCE_SPECS.items()
            },
            "materialized": bool(args.materialize),
            "release_variant": "source-linked reconstruction",
        }

        if args.materialize:
            archive_path = download_file(
                "MedXpertQA",
                SOURCE_SPECS["MedXpertQA"]["images_file"],
                cache_dir,
            )
            with zipfile.ZipFile(archive_path) as archive:
                members = build_zip_member_map(archive)
                examples = [
                    standardize_example(
                        locator,
                        matched[str(locator["medqa_mm_id"])],
                        output_dir,
                        cache_dir,
                        archive,
                        members,
                    )
                    for locator in locators
                ]
            write_jsonl(output_dir / "medqa_mm.jsonl", examples)
            summary["image_file_count"] = sum(len(row["image_paths"]) for row in examples)

        (output_dir / "validation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ReconstructionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
