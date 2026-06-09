from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from mcq_inference.common import MCQExample


ROOT = Path(__file__).resolve().parents[1]

SCHEME_OFFICIAL_MULTIMODAL = "official_multimodal_mcq"
SCHEME_MULTIMODAL_GE4 = "multimodal_ge4_options"
SCHEME_COMPACT_GE4_CAP2000 = "compact_ge4_cap2000_singleanswer"

SCHEME_CHOICES = (
    SCHEME_OFFICIAL_MULTIMODAL,
    SCHEME_MULTIMODAL_GE4,
    SCHEME_COMPACT_GE4_CAP2000,
)
DEFAULT_SCHEME = SCHEME_OFFICIAL_MULTIMODAL
COMPACT_SAMPLE_CAP = 2000
COMPACT_SAMPLE_SEED = 7

HEALTH_MEDICINE_SUBJECTS = {
    "Basic_Medical_Science",
    "Clinical_Medicine",
    "Diagnostics_and_Laboratory_Medicine",
    "Pharmacy",
    "Public_Health",
}


def manifest_root_for_scheme(scheme: str) -> Path:
    if scheme not in SCHEME_CHOICES:
        raise KeyError(f"Unsupported manifest scheme: {scheme}")
    return ROOT / "artifacts" / "manifests" / scheme


def dataset_manifest_dir(dataset: str, scheme: str = DEFAULT_SCHEME) -> Path:
    return manifest_root_for_scheme(scheme) / dataset


def _answer_letters(answer: Any) -> list[str]:
    text = str(answer or "").strip().upper()
    if not text:
        return []
    if "," in text:
        parts = [part.strip() for part in text.split(",")]
    else:
        parts = [text]
    letters = [part for part in parts if len(part) == 1 and part.isalpha()]
    return letters


def is_single_answer_value(answer: Any) -> bool:
    return len(_answer_letters(answer)) == 1 and "," not in str(answer or "")


def has_valid_answer(answer: Any) -> bool:
    return bool(_answer_letters(answer))


def joint_option_blocks(metadata: dict[str, Any]) -> list[list[str]]:
    blocks = []
    for block in metadata.get("joint_option_blocks") or []:
        clean_block = [str(item).strip() for item in block if str(item).strip()]
        if clean_block:
            blocks.append(clean_block)
    return blocks


def effective_option_count_from_parts(options: dict[str, str], metadata: dict[str, Any]) -> int | None:
    if options:
        return len(options)
    blocks = joint_option_blocks(metadata)
    if blocks:
        return min(len(block) for block in blocks)
    return None


def example_effective_option_count(example: MCQExample) -> int | None:
    return effective_option_count_from_parts(example.options, example.metadata)


def record_effective_option_count(record: dict[str, Any]) -> int | None:
    metadata = dict(record.get("metadata") or {})
    if metadata.get("effective_option_count") is not None:
        return int(metadata["effective_option_count"])
    return effective_option_count_from_parts(dict(record.get("options") or {}), metadata)


def has_explicit_options(example: MCQExample) -> bool:
    return example_effective_option_count(example) is not None


def has_any_local_image(example: MCQExample) -> bool:
    return bool(example.image_refs)


def include_official_split(dataset: str, metadata: dict[str, Any]) -> bool:
    if dataset == "medthinkvqa":
        return metadata.get("split") == "test"
    if dataset == "drvd_bench":
        return True
    if dataset == "3d_rad":
        return True
    if dataset == "mmmu_health_medicine":
        return metadata.get("split") == "test"
    if dataset == "mmmu_pro_health_medicine":
        return (
            metadata.get("split") == "test"
            and metadata.get("subset") == "standard (10 options)"
            and metadata.get("subject") in HEALTH_MEDICINE_SUBJECTS
        )
    if dataset == "omnimedvqa":
        return True
    if dataset == "medxpertqa_mm":
        return metadata.get("split") == "test"
    if dataset == "pmc_vqa":
        return metadata.get("source_file") == "test_clean.csv"
    if dataset == "cxreasonbench":
        return metadata.get("split") == "test"
    raise KeyError(f"Unsupported dataset key: {dataset}")


def example_matches_scheme(example: MCQExample, scheme: str) -> bool:
    if scheme not in SCHEME_CHOICES:
        raise KeyError(f"Unsupported manifest scheme: {scheme}")
    if not include_official_split(example.dataset, example.metadata):
        return False
    if not has_any_local_image(example):
        return False
    if not has_explicit_options(example):
        return False

    option_count = example_effective_option_count(example)
    if option_count is None:
        return False
    if scheme in {SCHEME_MULTIMODAL_GE4, SCHEME_COMPACT_GE4_CAP2000} and option_count < 4:
        return False
    if scheme == SCHEME_COMPACT_GE4_CAP2000 and not is_single_answer_value(example.correct_answer):
        return False
    return True


def augment_metadata_for_manifest(example: MCQExample, scheme: str) -> dict[str, Any]:
    option_count = example_effective_option_count(example)
    return {
        **example.metadata,
        "manifest_scheme": scheme,
        "effective_option_count": option_count,
        "has_local_image": has_any_local_image(example),
        "has_explicit_options": has_explicit_options(example),
        "has_valid_answer": has_valid_answer(example.correct_answer),
        "is_single_answer": is_single_answer_value(example.correct_answer),
    }


def compact_stratum_key(example: MCQExample) -> str:
    metadata = example.metadata
    option_count = example_effective_option_count(example) or 0
    if example.dataset == "drvd_bench":
        stage = metadata.get("task") or metadata.get("level") or metadata.get("module") or "unknown"
        return f"{metadata.get('module', 'unknown')}|{metadata.get('modality', 'unknown')}|{stage}|{option_count}"
    if example.dataset == "3d_rad":
        return f"{metadata.get('task_dir', 'unknown')}|{option_count}"
    if example.dataset == "mmmu_health_medicine":
        return str(metadata.get("subset") or "unknown")
    if example.dataset == "mmmu_pro_health_medicine":
        return str(metadata.get("subject") or "unknown")
    if example.dataset == "omnimedvqa":
        return (
            f"{metadata.get('source_dataset', 'unknown')}|"
            f"{metadata.get('modality_type', 'unknown')}|{option_count}"
        )
    if example.dataset == "cxreasonbench":
        return f"{metadata.get('task', 'unknown')}|{metadata.get('eval_path', 'unknown')}|{option_count}"
    return example.dataset


def _stable_example_rank(example: MCQExample, seed: int) -> tuple[str, str]:
    digest = hashlib.sha1(f"{seed}:{example.example_id}".encode("utf-8")).hexdigest()
    return digest, example.example_id


def _allocate_group_quotas(group_sizes: dict[str, int], cap: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    if total <= cap:
        return dict(group_sizes)

    quotas = {group: math.floor(size * cap / total) for group, size in group_sizes.items()}
    remaining = cap - sum(quotas.values())
    if remaining <= 0:
        return quotas

    ranked = sorted(
        group_sizes,
        key=lambda group: (
            -((group_sizes[group] * cap / total) - quotas[group]),
            -group_sizes[group],
            group,
        ),
    )
    for group in ranked[:remaining]:
        quotas[group] += 1
    return quotas


def deterministic_compact_sample(
    examples: Iterable[MCQExample],
    *,
    cap: int = COMPACT_SAMPLE_CAP,
    seed: int = COMPACT_SAMPLE_SEED,
) -> list[MCQExample]:
    items = list(examples)
    if len(items) <= cap:
        return sorted(items, key=lambda example: example.example_id)

    groups: dict[str, list[MCQExample]] = defaultdict(list)
    for example in items:
        groups[compact_stratum_key(example)].append(example)

    quotas = _allocate_group_quotas({group: len(values) for group, values in groups.items()}, cap)
    sampled: list[MCQExample] = []
    for group, values in groups.items():
        take = quotas.get(group, 0)
        ranked_values = sorted(values, key=lambda example: _stable_example_rank(example, seed))
        sampled.extend(ranked_values[:take])

    if len(sampled) != cap:
        ranked_all = sorted(items, key=lambda example: _stable_example_rank(example, seed))
        seen_ids = {example.example_id for example in sampled}
        for example in ranked_all:
            if len(sampled) >= cap:
                break
            if example.example_id in seen_ids:
                continue
            sampled.append(example)
            seen_ids.add(example.example_id)

    return sorted(sampled, key=lambda example: example.example_id)
