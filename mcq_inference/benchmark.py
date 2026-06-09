from __future__ import annotations

import csv
import io
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
from PIL import Image

from mcq_inference.common import (
    MCQExample,
    answer_letter_from_text,
    answer_text_from_options,
    clean_question_text,
    extract_answer_from_option_text,
    normalize_option_value,
    normalize_options,
    render_array_to_png,
    save_bytes_as_supported_image,
    strip_answer_choices_block,
    write_hf_image_value,
)
from mcq_inference.manifest_schemes import HEALTH_MEDICINE_SUBJECTS


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

MEDTHINKVQA_DIAGNOSIS_QUESTION = "What is the most likely diagnosis?"

DATASET_ORDER = [
    "medthinkvqa",
    "drvd_bench",
    "3d_rad",
    "mmmu_health_medicine",
    "mmmu_pro_health_medicine",
    "omnimedvqa",
    "medxpertqa_mm",
    "pmc_vqa",
    "cxreasonbench",
]


def safe_stem(text: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))
    sanitized = sanitized.strip("_")
    return sanitized[:160] or "example"


def reservoir_sample(items: Iterable[MCQExample], sample_size: int, seed: int) -> list[MCQExample]:
    rng = random.Random(seed)
    sampled: list[MCQExample] = []
    for index, item in enumerate(items):
        if index < sample_size:
            sampled.append(item)
            continue
        slot = rng.randint(0, index)
        if slot < sample_size:
            sampled[slot] = item
    return sampled


def iter_medthinkvqa_examples() -> Iterator[MCQExample]:
    dataset_root = DATA_DIR / "medthinkvqa"
    for split in ["train", "test"]:
        with (dataset_root / f"{split}.jsonl").open() as handle:
            for row_index, line in enumerate(handle):
                row = json.loads(line)
                options = normalize_options(row.get("options"))
                image_count = int(row.get("image_count") or 0)
                if not options or image_count <= 0:
                    continue
                image_refs = []
                for image_idx in range(1, image_count + 1):
                    image_path = row.get(f"image_{image_idx:02d}_path")
                    if image_path:
                        image_refs.append({"kind": "file", "path": str(dataset_root / image_path)})
                if not image_refs:
                    continue
                clinical_history = str(row.get("CLINICAL_HISTORY") or "").strip()
                if not clinical_history:
                    continue
                question = clean_question_text(
                    f"{MEDTHINKVQA_DIAGNOSIS_QUESTION}\n\nClinical history: {clinical_history}"
                )
                yield MCQExample(
                    dataset="medthinkvqa",
                    example_id=f"{split}_{row_index}_{row.get('title', 'case')}",
                    question=question,
                    options=options,
                    correct_answer=answer_letter_from_text(row.get("correct_answer")),
                    correct_answer_text=row.get("correct_answer_text"),
                    image_refs=image_refs,
                    metadata={
                        "split": split,
                        "title": row.get("title"),
                        "image_count": image_count,
                        "question_stem": MEDTHINKVQA_DIAGNOSIS_QUESTION,
                        "clinical_history": clinical_history,
                    },
                )


def iter_drvd_examples() -> Iterator[MCQExample]:
    dataset_root = DATA_DIR / "drvd_bench"

    with (dataset_root / "independent_qa.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            options = normalize_options(row.get("options"))
            image_path = dataset_root / row["image_paths"]
            if not options or not image_path.exists():
                continue
            answer = answer_letter_from_text(row.get("answer"))
            yield MCQExample(
                dataset="drvd_bench",
                example_id=f"independent_{row['unique_id']}_{row['level']}",
                question=clean_question_text(row["question"]),
                options=options,
                correct_answer=answer,
                correct_answer_text=answer_text_from_options(options, answer),
                image_refs=[{"kind": "file", "path": str(image_path)}],
                metadata={"module": "independent", "level": row["level"], "modality": row["modality"]},
            )

    with (dataset_root / "visual_evidence_qa.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            options = normalize_options(row.get("options"))
            image_path = dataset_root / row["image_paths"]
            if not options or not image_path.exists():
                continue
            answer = answer_letter_from_text(row.get("answer"))
            yield MCQExample(
                dataset="drvd_bench",
                example_id=f"visual_{row['unique_id']}_{row['task']}",
                question=clean_question_text(row["question"]),
                options=options,
                correct_answer=answer,
                correct_answer_text=answer_text_from_options(options, answer),
                image_refs=[{"kind": "file", "path": str(image_path)}],
                metadata={"module": "visual", "task": row["task"], "modality": row["modality"]},
            )

    with (dataset_root / "joint_qa.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            image_path = dataset_root / row["image_paths"]
            if not image_path.exists():
                continue
            letters = []
            option_blocks: list[list[str]] = []
            for field_name in ["modality", "organ", "lesion", "diagnosis"]:
                answer = answer_letter_from_text(row.get(f"{field_name}_answer"))
                options = [str(value).strip() for value in row.get(f"{field_name}_options", []) if str(value).strip()]
                if not answer or not options:
                    letters = []
                    break
                letters.append(answer)
                option_blocks.append(options)
            if not letters:
                continue
            yield MCQExample(
                dataset="drvd_bench",
                example_id=f"joint_{row['unique_id']}",
                question=str(row["joint_prompt"]).strip(),
                options={},
                correct_answer=",".join(letters),
                correct_answer_text=",".join(letters),
                image_refs=[{"kind": "file", "path": str(image_path)}],
                metadata={
                    "module": "joint",
                    "modality": row["modality"],
                    "response_type": "multi_letter_csv",
                    "expected_answer_count": len(letters),
                    "raw_prompt_includes_options": True,
                    "joint_option_blocks": option_blocks,
                },
            )


def _build_3d_rad_member_index() -> dict[str, list[str]]:
    series_members: dict[str, list[str]] = {}
    archive_path = DATA_DIR / "3d_rad" / "Images" / "test.zip"
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            folder_name = Path(member).parts[-2]
            series_members.setdefault(folder_name, []).append(member)
    for members in series_members.values():
        members.sort()
    return series_members


def iter_3d_rad_examples() -> Iterator[MCQExample]:
    archive_path = DATA_DIR / "3d_rad" / "Images" / "test.zip"
    series_members = _build_3d_rad_member_index()

    for csv_path in sorted((DATA_DIR / "3d_rad" / "test").rglob("*.csv")):
        task_name = csv_path.parent.name
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                if row.get("QuestionType") != "Close":
                    continue
                options = {}
                for letter in ["A", "B", "C", "D"]:
                    value = str(row.get(f"Choice {letter}", "")).strip()
                    if value and value != "-":
                        options[letter] = normalize_option_value(value)
                if not options:
                    continue
                volume_name = str(row["VolumeName"]).replace(".nii.gz", "")
                series_key = volume_name.rsplit("_", 1)[0]
                members = series_members.get(series_key, [])
                if not members:
                    continue
                answer = answer_letter_from_text(row.get("AnswerChoice"))
                yield MCQExample(
                    dataset="3d_rad",
                    example_id=f"{task_name}_{volume_name}_{row_index}",
                    question=clean_question_text(row["Question"]),
                    options=options,
                    correct_answer=answer,
                    correct_answer_text=normalize_option_value(row.get("Answer")),
                    image_refs=[
                        {
                            "kind": "zip",
                            "zip_path": str(archive_path),
                            "member": member,
                            "render_npy": True,
                        }
                        for member in members
                    ],
                    metadata={
                        "task_dir": task_name,
                        "volume_name": row["VolumeName"],
                        "source_csv": str(csv_path.relative_to(DATA_DIR)),
                    },
                )


def iter_mmmu_health_examples() -> Iterator[MCQExample]:
    dataset_root = DATA_DIR / "mmmu_health_medicine"
    image_cols = [f"image_{index}" for index in range(1, 8)]
    for parquet_path in sorted(dataset_root.glob("*/*.parquet")):
        subset = parquet_path.parent.name
        split = parquet_path.stem.split("-")[0]
        df = pd.read_parquet(
            parquet_path,
            columns=["id", "question", "options", "answer", "question_type", "subfield", *image_cols],
        )
        for row in df.itertuples(index=False):
            if row.question_type != "multiple-choice":
                continue
            options = normalize_options(row.options)
            image_refs = []
            for image_idx, col_name in enumerate(image_cols, start=1):
                image_value = getattr(row, col_name)
                if image_value is None:
                    continue
                image_refs.append(
                    {
                        "kind": "hf_image",
                        "value": image_value,
                        "filename": f"{getattr(row, 'id')}_image_{image_idx}",
                    }
                )
            if not options or not image_refs:
                continue
            answer = answer_letter_from_text(row.answer)
            yield MCQExample(
                dataset="mmmu_health_medicine",
                example_id=str(getattr(row, "id")),
                question=clean_question_text(str(getattr(row, "question"))),
                options=options,
                correct_answer=answer,
                correct_answer_text=answer_text_from_options(options, answer),
                image_refs=image_refs,
                metadata={"subset": subset, "split": split, "subfield": getattr(row, "subfield")},
            )


def iter_mmmu_pro_examples() -> Iterator[MCQExample]:
    from datasets import load_dataset

    subset_name = "standard (10 options)"
    dataset = load_dataset("MMMU/MMMU_Pro", subset_name, split="test")
    image_cols = [column for column in dataset.column_names if str(column).startswith("image_")]
    for row in dataset:
        subject = str(row.get("subject") or "").strip()
        if subject not in HEALTH_MEDICINE_SUBJECTS:
            continue
        options = normalize_options(row.get("options"))
        image_refs = []
        for image_idx, col_name in enumerate(image_cols, start=1):
            image_value = row.get(col_name)
            if image_value is None:
                continue
            image_refs.append(
                {
                    "kind": "hf_image",
                    "value": image_value,
                    "filename": f"{row.get('id')}_image_{image_idx}",
                }
            )
        if not options or not image_refs:
            continue
        answer = answer_letter_from_text(row.get("answer"))
        yield MCQExample(
            dataset="mmmu_pro_health_medicine",
            example_id=str(row.get("id")),
            question=clean_question_text(str(row.get("question") or "")),
            options=options,
            correct_answer=answer,
            correct_answer_text=answer_text_from_options(options, answer),
            image_refs=image_refs,
            metadata={
                "split": "test",
                "subset": subset_name,
                "subject": subject,
                "topic_difficulty": row.get("topic_difficulty"),
                "img_type": row.get("img_type"),
            },
        )


def iter_omnimedvqa_examples() -> Iterator[MCQExample]:
    archive_path = DATA_DIR / "omnimedvqa" / "OmniMedVQA.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        json_members = sorted(name for name in members if name.endswith(".json") and "QA_information" in name)
        for json_member in json_members:
            with archive.open(json_member) as handle:
                rows = json.load(handle)
            for row in rows:
                options = {}
                for key, value in row.items():
                    if key.startswith("option_") and str(value).strip():
                        letter = key.split("_", 1)[1].upper()
                        options[letter] = normalize_option_value(value)
                image_member = f"OmniMedVQA/{row['image_path']}"
                if not options or image_member not in members:
                    continue
                answer = extract_answer_from_option_text(options, row.get("gt_answer"))
                yield MCQExample(
                    dataset="omnimedvqa",
                    example_id=str(row["question_id"]),
                    question=clean_question_text(str(row["question"])),
                    options=options,
                    correct_answer=answer,
                    correct_answer_text=normalize_option_value(row.get("gt_answer")),
                    image_refs=[{"kind": "zip", "zip_path": str(archive_path), "member": image_member}],
                    metadata={
                        "question_type": row.get("question_type"),
                        "source_dataset": row.get("dataset"),
                        "modality_type": row.get("modality_type"),
                        "source_json": json_member,
                        "access_scope": "open_access_subset",
                    },
                )


def iter_medxpert_examples() -> Iterator[MCQExample]:
    archive_path = DATA_DIR / "medxpertqa_mm" / "images.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        for jsonl_path in [
            DATA_DIR / "medxpertqa_mm" / "MM" / "dev.jsonl",
            DATA_DIR / "medxpertqa_mm" / "MM" / "test.jsonl",
        ]:
            split = jsonl_path.stem
            with jsonl_path.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    options = normalize_options(row.get("options"))
                    image_refs = []
                    for image_name in row.get("images", []):
                        member = f"images/{image_name}"
                        if member in members:
                            image_refs.append({"kind": "zip", "zip_path": str(archive_path), "member": member})
                    if not options or not image_refs:
                        continue
                    answer = answer_letter_from_text(row.get("label"))
                    yield MCQExample(
                        dataset="medxpertqa_mm",
                        example_id=str(row["id"]),
                        question=strip_answer_choices_block(str(row["question"])),
                        options=options,
                        correct_answer=answer,
                        correct_answer_text=answer_text_from_options(options, answer),
                        image_refs=image_refs,
                        metadata={
                            "split": split,
                            "medical_task": row.get("medical_task"),
                            "body_system": row.get("body_system"),
                            "question_type": row.get("question_type"),
                        },
                    )


def iter_pmc_vqa_examples() -> Iterator[MCQExample]:
    dataset_root = DATA_DIR / "pmc_vqa"
    archive_v1 = dataset_root / "images.zip"
    archive_v2 = dataset_root / "images_2.zip"
    with zipfile.ZipFile(archive_v1) as zf_v1, zipfile.ZipFile(archive_v2) as zf_v2:
        members_v1 = set(zf_v1.namelist())
        members_v2 = set(zf_v2.namelist())

        for csv_name in ["train.csv", "test.csv", "test_clean.csv"]:
            with (dataset_root / csv_name).open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row_index, row in enumerate(reader):
                    member = f"images/{row['Figure_path']}"
                    if member not in members_v1:
                        continue
                    options = {
                        letter: normalize_option_value(row[f"Choice {letter}"])
                        for letter in ["A", "B", "C", "D"]
                        if row.get(f"Choice {letter}")
                    }
                    if not options:
                        continue
                    answer = answer_letter_from_text(row.get("Answer_label"))
                    yield MCQExample(
                        dataset="pmc_vqa",
                        example_id=f"{csv_name}_{row_index}",
                        question=clean_question_text(str(row["Question"])),
                        options=options,
                        correct_answer=answer,
                        correct_answer_text=normalize_option_value(row.get("Answer")),
                        image_refs=[{"kind": "zip", "zip_path": str(archive_v1), "member": member}],
                        metadata={"source_file": csv_name, "version": "v1"},
                    )

        for csv_name in ["train_2.csv", "test_2.csv"]:
            with (dataset_root / csv_name).open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row_index, row in enumerate(reader):
                    member = f"figures/{row['Figure_path']}"
                    if member not in members_v2:
                        continue
                    options = {
                        letter: normalize_option_value(row[f"Choice {letter}"])
                        for letter in ["A", "B", "C", "D"]
                        if row.get(f"Choice {letter}")
                    }
                    if not options:
                        continue
                    question = str(row.get("Question") or "").strip()
                    answer = answer_letter_from_text(row.get("Answer"))
                    yield MCQExample(
                        dataset="pmc_vqa",
                        example_id=f"{csv_name}_{row_index}",
                        question=clean_question_text(question),
                        options=options,
                        correct_answer=answer,
                        correct_answer_text=answer_text_from_options(options, answer),
                        image_refs=[{"kind": "zip", "zip_path": str(archive_v2), "member": member}],
                        metadata={"source_file": csv_name, "version": "v2"},
                    )


def iter_cxreasonbench_examples() -> Iterator[MCQExample]:
    manifest_path = DATA_DIR / "cxreasonbench" / "preprocessed" / "examples.jsonl"
    with manifest_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            options = {str(k): str(v) for k, v in (row.get("options") or {}).items()}
            if not options:
                continue
            image_paths = [str(Path(path).resolve()) for path in row.get("image_paths", [])]
            yield MCQExample(
                dataset="cxreasonbench",
                example_id=str(row["example_id"]),
                question=clean_question_text(str(row["question"])),
                options=options,
                correct_answer=row.get("correct_answer"),
                correct_answer_text=row.get("correct_answer_text"),
                image_refs=[{"kind": "file", "path": path} for path in image_paths],
                metadata=dict(row.get("metadata") or {}),
            )


DATASET_ITERATORS = {
    "medthinkvqa": iter_medthinkvqa_examples,
    "drvd_bench": iter_drvd_examples,
    "3d_rad": iter_3d_rad_examples,
    "mmmu_health_medicine": iter_mmmu_health_examples,
    "mmmu_pro_health_medicine": iter_mmmu_pro_examples,
    "omnimedvqa": iter_omnimedvqa_examples,
    "medxpertqa_mm": iter_medxpert_examples,
    "pmc_vqa": iter_pmc_vqa_examples,
    "cxreasonbench": iter_cxreasonbench_examples,
}


def materialize_example_images(example: MCQExample, image_root: Path) -> list[Path]:
    written_paths: list[Path] = []
    image_root.mkdir(parents=True, exist_ok=True)
    file_stem = safe_stem(example.example_id)
    for image_index, image_ref in enumerate(example.image_refs, start=1):
        base = image_root / f"{file_stem}_img{image_index}"
        kind = image_ref["kind"]
        if kind == "file":
            source = Path(image_ref["path"])
            suffix = (source.suffix or ".png").lower()
            if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                destination = base.with_suffix(source.suffix or ".png")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            else:
                destination = base.with_suffix(".png")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as image:
                    image.convert("RGB").save(destination)
            written_paths.append(destination)
            continue

        if kind == "zip":
            archive_path = Path(image_ref["zip_path"])
            member = image_ref["member"]
            with zipfile.ZipFile(archive_path) as archive:
                raw = archive.read(member)
            if image_ref.get("render_npy") or str(member).endswith(".npy"):
                destination = base.with_suffix(".png")
                render_array_to_png(raw, destination)
            else:
                destination = save_bytes_as_supported_image(
                    raw,
                    base,
                    preferred_suffix=Path(member).suffix or ".png",
                )
            written_paths.append(destination)
            continue

        if kind == "hf_image":
            destination = write_hf_image_value(image_ref["value"], base)
            written_paths.append(destination)
            continue

        raise ValueError(f"Unsupported image ref kind: {kind}")
    return written_paths


def build_smoke_set(output_dir: Path, per_dataset: int, seed: int) -> dict[str, int]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_root = output_dir / "images"
    manifest_path = output_dir / "examples.jsonl"

    summary: dict[str, int] = {}
    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for dataset_index, dataset_name in enumerate(DATASET_ORDER):
            iterator = DATASET_ITERATORS[dataset_name]()
            sampled = reservoir_sample(iterator, per_dataset, seed + dataset_index * 1009)
            sampled.sort(key=lambda example: example.example_id)
            summary[dataset_name] = len(sampled)

            for example in sampled:
                materialized = materialize_example_images(example, images_root / dataset_name)
                record = {
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "question": example.question,
                    "options": example.options,
                    "correct_answer": example.correct_answer,
                    "correct_answer_text": example.correct_answer_text,
                    "image_paths": [
                        str(path.relative_to(output_dir))
                        for path in materialized
                    ],
                    "metadata": example.metadata,
                }
                manifest_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
