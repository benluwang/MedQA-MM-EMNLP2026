# MEDQA-MM Identifier Release

This branch contains an identifier-only release of **MEDQA-MM**, a 1,000-item shortcut-mitigated subset constructed from MedThinkVQA, MedXpertQA, and MMMU-HM. It intentionally does **not** include question text, answer choices, gold answers, images, or other dataset payloads. The purpose of this release is to let users who have obtained the original datasets reconstruct the exact MEDQA-MM subset by matching stable source identifiers.

## Files

- `dataset/medqa_mm_ids.json`: the 1,000 source locators for MEDQA-MM.
- `dataset/README.md`: this description and reconstruction guide.

## Counts

| Source dataset | Count |
|---|---:|
| MedThinkVQA | 207 |
| MedXpertQA | 450 |
| MMMU-HM | 343 |
| **Total** | 1000 |

## How to link records back to the original datasets

Each record in `medqa_mm_ids.json` has a `medqa_mm_id` plus source locator fields. Use `dataset`, `source_dataset`, `split`, and `example_id` as the primary key. The fallback fields are included to make the mapping auditable against full test manifests.

Recommended matching rules:

| MEDQA-MM `dataset` | Original source | Primary locator | Fallback locator |
|---|---|---|---|
| `MedThinkVQA` | MedThinkVQA test split | `source_dataset=medthinkvqa`, `split=test`, `example_id` | `row_index` and `case_number` |
| `MedXpertQA` | `luludandan/medbias`, `medxpertqa_mm` test split | `source_dataset=medxpertqa_mm`, `split=test`, `example_id` such as `MM-1861` | `row_index` |
| `MMMU-HM` | `luludandan/medbias`, `mmmu_health_medicine` test split | `source_dataset=mmmu_health_medicine`, `split=test`, `example_id` such as `test_Clinical_Medicine_199` | `mmmu_subset` and `row_index` |

For MMMU-HM, `row_index` is only unique within a subject/subset, so the fallback locator uses both `mmmu_subset` and `row_index`.

## Validation

Before committing this branch, the 1,000 locators were checked against the complete local source manifests used to build the benchmark:

| Source manifest | Full manifest size | MEDQA-MM matches |
|---|---:|---:|
| MedThinkVQA test manifest | 720 | 207 |
| MedXpertQA-MM test manifest | 2000 | 450 |
| MMMU health/medicine test manifest | 1712 | 343 |

All 1,000 records matched exactly one source record by `example_id`. They also matched exactly one source record by fallback locator: `row_index` for MedThinkVQA and MedXpertQA, and `mmmu_subset + row_index` for MMMU-HM. The JSON was checked to ensure that payload fields such as `question`, `options`, `correct_answer`, `image_paths`, and `image_refs` are absent.
