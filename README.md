# MedQA-MM: Shortcuts Behind Medical Visual Reasoning

This is the official repository for the EMNLP 2026 paper **“MedQA-MM: Shortcuts Behind Medical Visual Reasoning.”**

It provides the released reference code, dataset locators, and core shortcut-auditing methods described in the paper. The repository currently contains the public release subset of the project; additional code, experiment logs, generated repair artifacts, aggregate results, and documentation will be added as they are prepared for release.

**Public dataset:** [🤗 bio-nlp-umass/MedQA-MM](https://huggingface.co/datasets/bio-nlp-umass/MedQA-MM)

## MedQA-MM Dataset Release

The public Hugging Face release contains the complete 1,000-item ID list: 207 MedThinkVQA locators, 450 MedXpertQA-MM locators, and 343 MMMU Health/Medicine locators. It is identifier-only and does not redistribute source questions, answer choices, labels, clinical text, or images.

Pinned source revisions and [`dataset/reconstruct_medqa_mm.py`](dataset/reconstruct_medqa_mm.py) let users validate every locator and, after accepting the upstream terms, build a local source-linked evaluation copy:

```bash
python -m pip install -r dataset/requirements.txt
python dataset/reconstruct_medqa_mm.py \
  --output-dir medqa_mm_local \
  --materialize \
  --accept-source-terms
```

The reconstructed payload is for local evaluation under the three upstream sources' terms and must not be redistributed. See [`dataset/README.md`](dataset/README.md) for the source revisions, licenses, restrictions, schema, and validation procedure.

## Repository Contents

- `dataset/`
  - Public identifier-only release files for MedQA-MM, mirrored at [Hugging Face](https://huggingface.co/datasets/bio-nlp-umass/MedQA-MM).
  - Provides the complete ID list, source revisions, release documentation, and a local reconstruction/validation script without redistributing external dataset payloads.

- `mcq_inference/`
  - Utilities for running multiple-choice inference experiments.
  - Includes manifest schemas, benchmark loading helpers, common normalization/parsing logic, and the package entrypoint for MCQ inference.

- `model/`
  - Lightweight model wrappers used by the inference utilities.
  - Includes interfaces for API-based and local/open-model backends such as GPT, Claude, Gemini, Together, and vLLM-style models.

- `shortcut/`
  - Core method code for the shortcut analyses discussed in the paper.
  - These files implement the main audit, repair, adjudication, and summarization logic for the publicly released shortcut families.

## Shortcut Method Files

- `shortcut/longest.py`
  - Length-gap shortcut analysis.
  - Selects items where the gold answer is uniquely longest, builds repair requests that lengthen distractors while preserving the correct answer, validates repaired options, and prepares paired manifests for evaluation.

- `shortcut/natural_cue.py`
  - Natural clinical-context cue analysis.
  - Screens question text for social or demographic context cues, prepares neutralization requests, supports model-assisted cue-use judging, and summarizes repeated before/after evaluations.

- `shortcut/device_material_cue.py`
  - Device/material cue audit.
  - Provides candidate screening and LLM adjudication helpers for cases where visible devices, materials, implants, or procedural artifacts may make an answer shortcut-like.

- `shortcut/image_text_shortcut.py`
  - Visible image-text shortcut audit and repair.
  - Classifies cases where rendered text in the image may reveal or overly constrain the answer, prepares image-edit repair requests, and supports repair/readiness judging.

- `shortcut/image_hint_cue.py`
  - Artificial image-hint shortcut analysis.
  - Implements broad and strict image-hint review logic, cue localization, mask preparation, image-edit request construction, repair judging, cue-use judging, and repeated-evaluation summaries.

## Release Notes

This repository supports method inspection rather than full numerical reproduction. Some paper-level outputs, including complete generated manifests, edited images, model prediction logs, expert-review records, and aggregate result tables, are not included in the current public release.

Additional materials will be added as they are prepared for public release.
