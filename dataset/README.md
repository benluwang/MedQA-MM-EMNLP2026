---
pretty_name: MedQA-MM
license: other
language:
- en
task_categories:
- visual-question-answering
tags:
- medical
- multimodal
- multiple-choice
- benchmark
- identifier-only
size_categories:
- 1K<n<10K
configs:
- config_name: identifiers
  data_files:
  - split: test
    path: data/medqa_mm_ids.jsonl
---

# MedQA-MM Identifier Release

[Paper repository](https://github.com/benluwang/MedQA-MM-EMNLP2026) ·
[Hugging Face dataset](https://huggingface.co/datasets/bio-nlp-umass/MedQA-MM)

MedQA-MM is a 1,000-item shortcut-mitigated medical multimodal multiple-choice benchmark constructed from MedThinkVQA, MedXpertQA-MM, and the Health and Medicine portion of MMMU. This public release is intentionally **identifier-only**.

It does **not** contain source questions, answer choices, gold answers, images, clinical text, or repaired payloads. It provides stable source locators, pinned source revisions, and a reconstruction script so that users who have independently obtained the source datasets under their respective terms can reproduce a local evaluation copy.

## Released files

- `data/medqa_mm_ids.jsonl`: one viewer-friendly locator record per MEDQA-MM item.
- `medqa_mm_ids.json`: the original structured 1,000-record locator release and validation metadata.
- `source_revisions.json`: official source repositories, pinned revisions, subsets, and terms links.
- `reconstruct_medqa_mm.py`: validates all locators and optionally materializes a local source-linked evaluation copy.
- `requirements.txt`: minimal reconstruction dependencies.

## Composition

| Source dataset | Eligible candidate items | MEDQA-MM items |
|---|---:|---:|
| MedThinkVQA | 720 | 207 |
| MedXpertQA-MM | 2,000 | 450 |
| MMMU Health/Medicine | 1,712 | 343 |
| **Total** | **4,432** | **1,000** |

For MMMU Health/Medicine, the five source test subsets contain 1,752 rows; the 1,712 candidate count excludes 40 open-response rows and matches the multiple-choice eligibility rule used by the construction pipeline.

## Source repositories and terms

| MEDQA-MM source | Official repository | Pinned revision | Terms relevant to reconstruction |
|---|---|---|---|
| MedThinkVQA | [`bio-nlp-umass/MedThinkVQA`](https://huggingface.co/datasets/bio-nlp-umass/MedThinkVQA) | `5555b80e167c4796efab92fe709fb05914077dc0` | CC BY-NC-SA 4.0; research and education only; source card restrictions apply. |
| MedXpertQA-MM | [`TsinghuaC3I/MedXpertQA`](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) | `7e7c465a68eb2b866926bfa59c8c9d17a8daba65` | The paper states evaluation-only use and asks users not to share examples online. Do not redistribute the reconstructed payload. |
| MMMU Health/Medicine | [`MMMU/MMMU`](https://huggingface.co/datasets/MMMU/MMMU) | `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68` | Apache-2.0 dataset card; source-level copyright and removal disclaimer also applies. |

The upstream sources retain all rights in their questions, answers, images, and associated content. Publishing this locator list does not grant permission to redistribute any source payload.

## Locator schema

Each JSONL row contains:

- `medqa_mm_id`: stable MEDQA-MM identifier.
- `dataset`: one of `MedThinkVQA`, `MedXpertQA`, or `MMMU-HM`.
- `source_repo` and `source_revision`: official pinned source.
- `source_config`, `split`, `example_id`, and `row_index`: primary reconstruction locator.
- `case_number` or `mmmu_subset` where needed for fallback matching.
- `source_license_or_terms` and `source_terms_url`: source-specific release notice.

The original JSON additionally records the validation performed against the complete construction manifests.

## Validate the identifiers

From this directory:

```bash
python -m pip install -r requirements.txt
python reconstruct_medqa_mm.py \
  --output-dir medqa_mm_validation
```

The default mode downloads only the source metadata needed to resolve the locators. It writes:

- `resolved_ids.jsonl`
- `validation_summary.json`

Successful validation requires all 1,000 locators to match exactly and reproduces the 207/450/343 source counts.

## Create a local evaluation copy

Read all three source terms first. Then run:

```bash
python reconstruct_medqa_mm.py \
  --output-dir medqa_mm_local \
  --materialize \
  --accept-source-terms
```

This additionally creates:

- `medqa_mm.jsonl`: standardized local MCQ records.
- `images/MEDQA-MM-XXXX/`: locally materialized source images.

The output is for local evaluation under the upstream terms. **Do not upload or redistribute it.** It is a source-linked reconstruction and does not include unpublished repair artifacts that are not part of this identifier release.

## License and responsible use

This repository uses `license: other` because the release combines author-created locator metadata with references to independently governed source datasets. No license to the underlying source payloads is granted here.

MedQA-MM is a research benchmark, not a clinical device. It must not be used for diagnosis, treatment, triage, or other clinical decision-making. Users are responsible for reviewing the current upstream licenses, terms, privacy notices, and source revisions before reconstruction.

## Citation

```bibtex
@inproceedings{wang2026medqamm,
  title     = {MedQA-MM: Shortcuts Behind Medical Visual Reasoning},
  author    = {Wang, Benlu and Zhang, Yifan and Yu, Jiaqing and Ong, Chin Siang and Huang, Juncheng and Li, Zhuohao and Zhang, Zhenyu and Cohan, Arman and Yu, Hong and Yao, Zonghai},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

Please also cite the three original source datasets when using a reconstructed local copy.
