# MedQA-MM: Shortcuts Behind Medical Visual Reasoning

This is the official repository for the EMNLP 2026 paper **“MedQA-MM: Shortcuts Behind Medical Visual Reasoning.”**

It provides the released reference code, dataset locators, and core shortcut-auditing methods described in the paper. The repository currently contains the public release subset of the project; additional code, experiment logs, generated repair artifacts, aggregate results, and documentation will be added as they are prepared for release.

## Repository Contents

- `dataset/`
  - Identifier-only release files for MedQA-MM.
  - These files provide item locators and dataset metadata needed to identify the released benchmark items without redistributing external dataset payloads.

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
