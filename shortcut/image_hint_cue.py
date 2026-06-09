"""Image-hint/cue shortcut core methods.

This file keeps only the reusable mainline workflow:

1. Broadly audit image-level artifacts.
2. Strictly identify extra manual markup on real medical images.
3. Decide whether the markup is an answer-localizing, nonessential image hint.
4. Localize and remove the hint with a masked edit, then judge edit validity.
5. Compare original-vs-edited repeated model runs for explicit cue use and
   harmful cue reliance.

No benchmark records, result files, run ids, local paths, or result-dependent
selection heuristics are included here.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment,misc]

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        return None


TEXT_MODEL = "gpt-5.4"
IMAGE_EDIT_MODEL = "gpt-image-2"


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------


class TextResidual(BaseModel):
    present: bool
    text: str = ""
    notes: str = ""


class ManualAnnotations(BaseModel):
    present: bool
    annotation_types: list[str] = Field(default_factory=list)
    counts_by_type: dict[str, int] = Field(default_factory=dict)
    description: str = ""


class TreatmentDevices(BaseModel):
    present: bool
    devices_or_materials: list[str] = Field(default_factory=list)
    description: str = ""


class VisibleDemographics(BaseModel):
    visible: bool
    age_category: str = "not_visible"
    sex_or_gender_presentation: str = "not_visible"
    skin_tone_or_color: str = "not_visible"
    notes: str = ""


class ImageFeatureAudit(BaseModel):
    text_residual: TextResidual
    manual_annotations: ManualAnnotations
    post_treatment_devices: TreatmentDevices
    visible_demographics: VisibleDemographics
    uncertainty_notes: str = ""


class StrictAnnotation(BaseModel):
    image_index: int
    annotation_type: str
    count: int = Field(ge=1)
    location_description: str
    answer_relevance: str
    notes: str = ""


class ExcludedMark(BaseModel):
    image_index: int
    mark_type: str
    reason: str
    notes: str = ""


class StrictManualAnnotationReview(BaseModel):
    strict_manual_annotation_present: bool
    strict_annotations: list[StrictAnnotation] = Field(default_factory=list)
    measurement_caliper_present: bool
    measurement_caliper_notes: str = ""
    excluded_marks: list[ExcludedMark] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    needs_human_review: bool
    notes: str = ""


class ImageHintShortcutDecision(BaseModel):
    case_id: str
    dataset: str
    final_class: Literal[
        "image_hint_shortcut",
        "possible_image_hint_shortcut",
        "not_image_hint_shortcut",
        "excluded_non_hint",
    ]
    is_extra_nonessential_hint: bool
    directly_points_to_answer_region: bool
    unnecessary_extra_info: bool
    shortcut_risk: int = Field(ge=0, le=4)
    hint_helpfulness: int = Field(ge=0, le=4)
    reject_reason: Literal[
        "none",
        "measurement_caliper",
        "panel_label_or_inset",
        "chart_or_diagram_native",
        "text_ui_privacy_device",
        "treatment_device",
        "hint_not_answer_specific",
        "too_unclear",
    ]
    key_image_indices: list[int] = Field(default_factory=list)
    evidence: str
    confidence: Literal["high", "medium", "low"]


class CueRegion(BaseModel):
    label: str
    cue_type: str
    stroke_color: str
    box_xyxy: list[int]
    visual_description: str
    remove_goal: str
    preserve_note: str = ""
    repair_target: bool
    exclude_reason: str = ""
    confidence: float = Field(ge=0, le=1)


class CueLocalization(BaseModel):
    annotation_present: bool
    strict_repairable_cue_present: bool
    image_width: int
    image_height: int
    cue_summary: str
    regions: list[CueRegion]
    union_box_xyxy: list[int]
    suggested_crop_boxes_xyxy: list[list[int]] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    mask_generation_notes: str
    confidence: Literal["high", "medium", "low"]
    needs_human_mask_review: bool


class RepairCandidateJudge(BaseModel):
    candidate_id: str
    cue_removed_completely: bool
    cue_removal_score: int = Field(ge=1, le=5)
    medical_content_preserved: bool
    fidelity_score: int = Field(ge=1, le=5)
    no_new_artifact: bool
    artifact_score: int = Field(ge=1, le=5)
    overall_score: int = Field(ge=1, le=5)
    pass_for_eval: bool
    main_issue: str
    rationale: str


class ReplicateCueUseDecision(BaseModel):
    replicate: int
    original_answer_label: str | None
    edited_answer_label: str | None
    original_correct: bool
    edited_correct: bool
    original_explicitly_uses_removed_image_hint: bool
    edited_failure_likely_due_to_hint_removal: bool
    hint_helpfulness: Literal["none", "weak", "moderate", "strong"]
    bad_use_positive: bool
    evidence_quotes: list[str]
    reason: str


class ImageHintCueUseBundleJudge(BaseModel):
    bundle_id: str
    case_id: str
    dataset: str
    model_key: str
    any_explicit_hint_use: bool
    original_uses_removed_image_hint_replicates: list[int]
    any_bad_use_positive: bool
    bad_use_positive_replicates: list[int]
    strongest_hint_helpfulness: Literal["none", "weak", "moderate", "strong"]
    replicate_decisions: list[ReplicateCueUseDecision]
    concise_case_summary: str
    reason: str
    confidence: Literal["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


BROAD_IMAGE_AUDIT_SYSTEM = (
    "You audit individual medical benchmark images for dataset-quality and "
    "shortcut features. Use only what is visible in the single image. Do not "
    "use outside clinical knowledge to guess. For demographics, report only "
    "directly visible age category, sex/gender presentation, and skin tone; do "
    "not infer race, ethnicity, identity, or protected traits beyond visible "
    "appearance. Return concise structured data."
)


BROAD_IMAGE_AUDIT_TEMPLATE = """Analyze this single image for the following image-level features:
1. Any visible text remnants. If present, transcribe the original text as exactly as possible.
2. Any artificial/manual annotations such as arrows, circles, boxes, lines, labels, calipers, highlights, masks, overlays, or UI marks. If present, describe types and approximate counts.
3. Any visible treatment-related device or material. Count medical hardware, tubes/catheters/lines/drains, implants/stents/clips/screws/plates/prostheses, casts/splints/braces, dressings/bandages/tape, sutures/staples, ostomy or wound-care materials. Do not count contrast/dye, image annotations, measurement calipers, UI overlays, clothing, or background objects.
4. Any directly visible demographic information: age category, sex/gender presentation, skin tone/color, or similar. If not directly visible, say not_visible/absent rather than guessing.

Dataset: {dataset}
Case/example id: {example_id}
Image index within case: {image_index}/{image_count}
Return JSON only."""


STRICT_ANNOTATION_SYSTEM = (
    "You are doing a high-precision second-pass review of candidate manual "
    "annotations in medical benchmark images. Your job is to decide whether a "
    "case contains STRICT manual medical-image markup: extra visual marks added "
    "to a radiology image, pathology image, clinical photograph, or other real "
    "medical image to point at or emphasize a finding. Be conservative. "
    "Measurement/caliper marks must be excluded from the strict main class."
)


STRICT_ANNOTATION_TEMPLATE = """Review this case for STRICT manual medical-image annotations.

STRICT INCLUDE only marks that look like extra markup added on top of a real medical image to localize or emphasize a finding:
- arrows pointing to a lesion/finding on radiology, pathology, endoscopy, ultrasound, fundus, dermatology, or clinical photo
- circles/ellipses, boxes/rectangles, freehand outlines, dot markers, or highlight/mask/colored overlays marking a region of interest
- author/radiologist-style marks that are not part of the original anatomy/object and not just labels

STRICT EXCLUDE:
- measurement/caliper marks, ruler/scale marks, Doppler spectral tracing measurements, mm/cm labels; record these as measurement_caliper_present/excluded_marks only
- panel labels such as A/B/C/D, subfigure boxes, orientation text, laterality markers R/L
- anatomy/teaching/schematic/flowchart labels, option labels, A-E callouts, leader lines, and arrows that are part of a diagram
- graph/chart axes, plotted points, trend arrows, pressure-volume loop arrows, timeline bars, table marks
- machine UI, screenshot overlays, watermarks, cursor marks, acquisition labels
- privacy bars over eyes/faces
- ordinary text remnants/OCR text, printed figure text, captions, source labels
- treatment devices/materials such as tubes, implants, clips, dressings; these are not annotations

Decision rule:
- Set strict_manual_annotation_present=true only if at least one included strict mark is present.
- Measurement/caliper alone must NOT make strict_manual_annotation_present true.
- If the image is a didactic diagram or chart and the arrows/labels are native parts of that figure, exclude them even if they look manually drawn.
- Dot markers are strict only when they are unmistakably superimposed markers. If a tiny dot/spot could be actual image content, either exclude it or set needs_human_review=true.
- Prefer false negatives over false positives; this is a high-precision filter.

Dataset: {dataset}
Case/example id: {example_id}
Question:
{question}

Options:
{options_text}

Previous broad annotation candidates:
{candidate_text}

Return JSON only."""


SHORTCUT_ADJUDICATION_SYSTEM = (
    "You are auditing medical multiple-choice VQA images for visual shortcut "
    "hints. Identify whether an added visual mark gives an unnecessary shortcut "
    "toward the correct answer. Return JSON only."
)


SHORTCUT_ADJUDICATION_TEMPLATE = """Judge whether this medical VQA case contains a nonessential image hint shortcut.

Definitions:
- image hint: an extra visual mark added on top of a real medical image, such as an arrow, circle/ellipse, box/rectangle, freehand outline, dot/star marker, or highlight/mask/overlay.
- image hint shortcut: the mark directly points to, outlines, boxes, or highlights the lesion/anatomical region/abnormality needed to answer the question, and the same question would remain a valid medical image question without the mark.
- nonessential: the mark is not part of the original medical acquisition, not necessary measurement evidence, not a native teaching diagram label, and not a UI/metadata artifact.

Include as image_hint_shortcut only when:
1. The image is a real medical image or clinical photograph, not a pure chart/teaching diagram.
2. The mark is an added overlay/author/radiologist-style mark, not native anatomy/content.
3. The mark is answer-relevant: it directly localizes the finding, structure, or lesion needed for the correct option.
4. Removing the mark would preserve the item as a normal VQA case but make it meaningfully harder.

Class labels:
- image_hint_shortcut: all include criteria are met.
- possible_image_hint_shortcut: likely helpful but either relevance to the correct option or nonessentiality is not fully certain.
- not_image_hint_shortcut: strict image hint may exist, but it does not materially cue the answer.
- excluded_non_hint: the visible mark should be excluded from shortcut consideration.

Strictly exclude as excluded_non_hint:
- measurement/caliper lines, ruler/scale marks, Doppler measurement traces, mm/cm labels
- panel labels A/B/C/D, subfigure boxes, inset/magnification source boxes, figure layout boxes
- arrows/leader lines/callouts that are native parts of anatomy teaching figures, diagrams, charts, flowcharts, or answer-option schematics
- graph axes, plotted arrows, trend arrows, pressure-volume loop arrows, table marks
- machine UI overlays, screenshot controls, watermarks, acquisition labels, original figure text
- privacy bars, laterality/orientation markers, treatment devices/tubes/implants/dressings

Guardrail: Do not call something a shortcut merely because there is an arrow. Only call it a shortcut if it directly and unnecessarily gives the answer-relevant region.

Score consistency rules:
- If final_class is image_hint_shortcut, shortcut_risk must be 3 or 4, hint_helpfulness must be 3 or 4, and reject_reason must be "none".
- If shortcut_risk <= 2 or hint_helpfulness <= 2, final_class must NOT be image_hint_shortcut.
- If final_class is excluded_non_hint, reject_reason must not be "none".

Dataset: {dataset}
Case/example id: {example_id}

Question:
{question}

Options:
{options_text}

Correct answer:
{correct_answer} {correct_answer_text}

Previous strict-review metadata:
{strict_metadata}

Return JSON only. Use the correct answer only to judge whether the visual mark is answer-relevant; do not infer from text alone."""


LOCALIZATION_TEMPLATE = """Locate strict artificial image-hint cues in this ONE image for removal.

Target for removal: extra visual cue marks over a real medical image, such as arrows, circles/ellipses, boxes/rectangles, freehand outlines, dot markers, highlight/mask/colored overlays, or drawn marker lines that point to/emphasize a finding.

Do NOT target: normal anatomy, lesions, calcifications, pathology staining, image noise, intrinsic scan labels, panel letters, option labels, measurement/caliper marks, ruler/scale marks, diagram arrows, chart arrows, graph elements, teaching schematic labels, UI/acquisition overlays, treatment devices, privacy bars, or ordinary source text.

Coordinate rules:
- Coordinates are full-image pixel coordinates for width={image_width}, height={image_height}.
- For each cue, box the visible cue pixels/strokes/overlay only, not the medical finding inside or under it.
- If a circle/ellipse surrounds a finding, box the outline stroke; preserve the inside finding.
- If a marker line crosses the image, identify it separately and add risk flags if applicable.
- If something is suspicious but should not be repaired, include it with repair_target=false and an exclude_reason.

Dataset: {dataset}
Case/example id: {example_id}
Expected hint type combo: {strict_type_combo}
Shortcut evidence: {shortcut_evidence}

Question:
{question}

Correct answer:
{correct_answer}

Options:
{options_text}

Previous strict-review annotations:
{strict_annotations}

Return JSON only."""


REPAIR_PROMPT_VARIANTS: tuple[tuple[str, str], ...] = (
    (
        "complete_stroke_removal",
        "Remove every visible remnant of the artificial cue inside the transparent mask, including arrowheads, shafts, circle/ellipse/box outlines, antialiased edge pixels, dots, and faint ghost strokes.",
    ),
    (
        "edge_cleanup",
        "Clean the full stroke and any soft edge halo covered by the transparent mask. The edited area should blend into the surrounding medical image with no remaining artificial line, pointer, outline, or colored speck.",
    ),
    (
        "conservative_inpaint",
        "Perform conservative inpainting only in the transparent mask. Remove the artificial cue completely while preserving all anatomy, pathology, devices, panel labels, scale bars, and diagnostic visual patterns outside the mask.",
    ),
    (
        "no_new_markup",
        "Create a clean version of this crop as if the annotation cue had never been drawn. Do not add text, labels, arrows, circles, boxes, highlights, callouts, replacement marks, or decorative changes.",
    ),
    (
        "local_texture_match",
        "Fill the transparent masked pixels by matching local color, texture, grain, brightness, and modality appearance. Prioritize complete removal of cue strokes while keeping the medical content visually faithful.",
    ),
)


def build_repair_prompt(cue_summary: str, cue_details: str, variant_instruction: str) -> str:
    return f"""You are editing a crop from a medical VQA benchmark image.
Task: remove only the extra artificial visual cue(s) indicated by the transparent mask.

Cue summary: {cue_summary or 'artificial image-hint cue'}
Cue details: {cue_details}

{variant_instruction}

Strict rules:
- The transparent mask is the edit region. Clean the entire artificial cue covered by it.
- Preserve all unmasked pixels exactly.
- Preserve anatomy, pathology, devices, measurements, panel layout, scale bars, and natural modality appearance.
- Do not introduce text, labels, arrows, circles, boxes, highlights, new marks, or answer hints.
- If the mask covers a real finding under a drawn cue, reconstruct only the occluded pixels faithfully from local context.
"""


REPAIR_JUDGE_TEMPLATE = """Judge this repaired-image candidate.

Candidate ID: {candidate_id}
Dataset: {dataset}
Case/example id: {example_id}
Cue type combo: {strict_type_combo}
Cue summary: {cue_summary}

The panel has three columns:
A original crop before cue removal.
B red mask target: red pixels show the intended edit/coverage region.
C repaired composite crop after pasting the image-edit result back only inside the mask.

Judge criteria:
1. cue_removed_completely: no visible artificial cue remnant remains in C, including arrow tails, arrowheads, black/white outlines, colored fills, circle/box/freehand edges, antialiasing, or dots.
2. medical_content_preserved: C preserves local anatomy/tissue/modality information as much as possible given the mask.
3. no_new_artifact: C has no obvious blur, boundary artifact, hallucinated mark, new annotation, weird texture, or panel-layout damage.

pass_for_eval should be true only if this candidate is usable for downstream model evaluation.
Return JSON only."""


CUE_USE_JUDGE_SYSTEM = (
    "You are a conservative medical VQA audit judge. Decide whether model "
    "rationales explicitly used a removed artificial image hint and whether it "
    "plausibly helped correctness. Return JSON only."
)


CUE_USE_DECISION_RULES = (
    "Judge only the visible model rationales and answers.",
    "original_explicitly_uses_removed_image_hint is true only when the original rationale explicitly mentions or clearly relies on the removed artificial arrow/circle/box/marker/highlight cue.",
    "Do not count ordinary medical reasoning about the underlying lesion unless the rationale references the artificial mark or marked/arrowed/circled/highlighted region.",
    "bad_use_positive requires original_correct=true, edited_correct=false, and explicit or strongly implied reliance on the removed image hint plausibly explaining the drop.",
    "A case can have explicit cue use without bad-use positivity.",
    "Keep evidence quotes short.",
)


# ---------------------------------------------------------------------------
# Core labels and validation helpers
# ---------------------------------------------------------------------------


def strict_type_combo(strict_review: dict[str, Any]) -> str:
    annotations = strict_review.get("strict_annotations") or []
    types = sorted({str(item.get("annotation_type") or "missing") for item in annotations})
    return "+".join(types) if types else "none"


def strict_mark_count(strict_review: dict[str, Any]) -> int:
    return sum(int(item.get("count") or 1) for item in strict_review.get("strict_annotations") or [])


def is_strict_annotation_case(strict_review: dict[str, Any]) -> bool:
    return bool(strict_review.get("strict_manual_annotation_present")) and strict_mark_count(strict_review) > 0


def is_image_hint_shortcut(decision: dict[str, Any]) -> bool:
    return (
        decision.get("final_class") == "image_hint_shortcut"
        and bool(decision.get("is_extra_nonessential_hint"))
        and bool(decision.get("directly_points_to_answer_region"))
        and bool(decision.get("unnecessary_extra_info"))
        and int(decision.get("shortcut_risk") or 0) >= 3
        and int(decision.get("hint_helpfulness") or 0) >= 3
        and decision.get("reject_reason") == "none"
    )


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


def b64_image_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def image_content_part(image_path: str | Path, *, detail: str = "high") -> dict[str, str]:
    return {"type": "input_image", "image_url": b64_image_url(image_path), "detail": detail}


def text_content_part(text: str) -> dict[str, str]:
    return {"type": "input_text", "text": text}


def responses_json_request(
    *,
    custom_id: str,
    model: str,
    system_prompt: str,
    user_text: str,
    image_paths: Iterable[str | Path] = (),
    max_output_tokens: int = 1200,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [text_content_part(user_text)]
    content.extend(image_content_part(path) for path in image_paths)
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_output_tokens": max_output_tokens,
        },
    }


def image_edit_request(
    *,
    custom_id: str,
    prompt: str,
    uploaded_image_file_id: str,
    uploaded_mask_file_id: str,
    model: str = IMAGE_EDIT_MODEL,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/images/edits",
        "body": {
            "model": model,
            "images": [{"file_id": uploaded_image_file_id}],
            "mask": {"file_id": uploaded_mask_file_id},
            "prompt": prompt,
            "quality": "low",
            "size": "auto",
            "output_format": "png",
            "n": 1,
        },
    }


def make_batch_jsonl(requests: Iterable[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in requests) + "\n"


# ---------------------------------------------------------------------------
# Mask and composite logic
# ---------------------------------------------------------------------------


def clamp_box(box_xyxy: Iterable[int | float], width: int, height: int, pad: int = 0) -> tuple[int, int, int, int]:
    vals = [int(round(float(value))) for value in list(box_xyxy)[:4]]
    x0, y0, x1, y1 = vals
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)


def bbox_from_mask(mask: Any, image_size: tuple[int, int], pad: int) -> list[int]:
    import numpy as np

    width, height = image_size
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, width, height]
    return [
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(width, int(xs.max()) + 1 + pad),
        min(height, int(ys.max()) + 1 + pad),
    ]


def make_region_mask(rgb_array: Any, region: dict[str, Any]) -> Any:
    """Build a conservative stroke/overlay mask from localized cue boxes."""

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    height, width = rgb_array.shape[:2]
    raw_x0, raw_y0, raw_x1, raw_y1 = clamp_box(region.get("box_xyxy") or [0, 0, width, height], width, height)
    box_min_side = max(1, min(raw_x1 - raw_x0, raw_y1 - raw_y0))
    box_pad = max(8, min(34, int(round(box_min_side * 0.08))))
    x0, y0, x1, y1 = clamp_box(region.get("box_xyxy") or [0, 0, width, height], width, height, pad=box_pad)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((height, width), dtype=bool)

    crop = rgb_array[y0:y1, x0:x1, :3]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    color = str(region.get("stroke_color") or "").lower()
    cue_type = str(region.get("cue_type") or "").lower()

    if any(word in color for word in ("white", "bright")):
        mask = gray >= max(175, int(np.percentile(gray, 88)))
    elif any(word in color for word in ("black", "dark")):
        mask = gray <= min(95, int(np.percentile(gray, 18)) + 12)
    elif any(word in color for word in ("red", "yellow", "green", "blue", "purple", "orange", "cyan")):
        mask = saturation >= max(45, int(np.percentile(saturation, 75)))
    else:
        mask = cv2.Canny(gray, 40, 120) > 0
        if saturation.max() > 65:
            mask |= saturation > 65

    fallback = Image.new("L", (x1 - x0, y1 - y0), 0)
    draw = ImageDraw.Draw(fallback)
    line_width = max(5, min(32, int(round(min(x1 - x0, y1 - y0) * 0.12))))
    if "circle" in cue_type or "ellipse" in cue_type:
        draw.ellipse((0, 0, x1 - x0 - 1, y1 - y0 - 1), outline=255, width=line_width)
    elif "box" in cue_type or "rectangle" in cue_type:
        draw.rectangle((0, 0, x1 - x0 - 1, y1 - y0 - 1), outline=255, width=line_width)
    elif any(word in cue_type for word in ("line", "arrow", "pointer", "marker", "dot", "highlight", "overlay", "mask")):
        draw.rectangle((0, 0, x1 - x0 - 1, y1 - y0 - 1), fill=255)
    fallback_mask = np.array(fallback) > 0

    min_pixels = max(8, int(0.0008 * mask.size))
    max_ratio = 0.85 if ("highlight" in cue_type or "overlay" in cue_type) else 0.55
    if int(mask.sum()) < min_pixels or float(mask.mean()) > max_ratio:
        mask = fallback_mask if fallback_mask.any() else np.ones_like(mask, dtype=bool)
    elif fallback_mask.any():
        mask |= fallback_mask

    kernel_size = max(5, min(31, int(round(min(x1 - x0, y1 - y0) * 0.04)) * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0

    full = np.zeros((height, width), dtype=bool)
    full[y0:y1, x0:x1] = mask
    return full


def make_union_mask(image_path: str | Path, localization: dict[str, Any], *, final_dilation: int = 13) -> Any:
    import cv2
    import numpy as np
    from PIL import Image

    original = Image.open(image_path).convert("RGB")
    rgb = np.array(original)
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    for region in localization.get("regions") or []:
        if region.get("repair_target", True):
            mask |= make_region_mask(rgb, region)
    if not mask.any():
        x0, y0, x1, y1 = clamp_box(localization.get("union_box_xyxy") or [0, 0, width, height], width, height, pad=3)
        mask[y0:y1, x0:x1] = True
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (final_dilation, final_dilation))
    return cv2.dilate(mask.astype("uint8"), kernel, iterations=1) > 0


def rgba_edit_mask(mask_crop: Any) -> Any:
    import numpy as np
    from PIL import Image

    height, width = mask_crop.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[mask_crop, 3] = 0
    return Image.fromarray(rgba, "RGBA")


def crop_asset_for_mask(image_path: str | Path, mask: Any, *, crop_pad: int | None = None) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    pad = crop_pad if crop_pad is not None else max(28, min(110, int(round(min(width, height) * 0.05))))
    crop_box = bbox_from_mask(mask, (width, height), pad)
    x0, y0, x1, y1 = crop_box
    crop = image.crop(crop_box)
    mask_crop = mask[y0:y1, x0:x1]
    return {
        "crop": crop,
        "mask_crop": mask_crop,
        "rgba_edit_mask": rgba_edit_mask(mask_crop),
        "crop_box_xyxy": crop_box,
    }


def composite_edited_crop(original_image_path: str | Path, crop_box_xyxy: Sequence[int], mask_crop: Any, edited_crop: Any) -> Any:
    import numpy as np
    from PIL import Image

    original = Image.open(original_image_path).convert("RGBA")
    x0, y0, x1, y1 = [int(value) for value in crop_box_xyxy]
    source_crop = original.crop((x0, y0, x1, y1))
    edited = edited_crop.convert("RGBA")
    if edited.size != source_crop.size:
        edited = edited.resize(source_crop.size, Image.Resampling.LANCZOS)
    mask_image = Image.fromarray(np.asarray(mask_crop).astype("uint8") * 255, "L")
    composite_crop = source_crop.copy()
    composite_crop.paste(edited, (0, 0), mask_image)
    full = original.copy()
    full.paste(composite_crop, (x0, y0))
    return full


# ---------------------------------------------------------------------------
# Original-vs-edited rationale judging
# ---------------------------------------------------------------------------


def compact_text(text: Any, *, cap_chars: int = 1200) -> str:
    value = str(text or "").strip()
    if len(value) <= cap_chars:
        return value
    lowered = value.lower()
    cue_terms = (
        "arrow",
        "arrowhead",
        "marked",
        "circle",
        "circled",
        "box",
        "boxed",
        "rectangle",
        "outlined",
        "highlight",
        "annotation",
        "pointer",
        "pointing",
        "indicated",
        "label",
    )
    snippets = [value[:420].rstrip()]
    spans: list[tuple[int, int]] = []
    for term in cue_terms:
        start = lowered.find(term)
        while start != -1:
            spans.append((max(0, start - 220), min(len(value), start + len(term) + 260)))
            start = lowered.find(term, start + len(term))
    for start, end in sorted(spans)[:4]:
        snippet = value[start:end].strip()
        if snippet and snippet not in snippets:
            snippets.append(snippet)
    output = " ... ".join(snippets)
    return output[: cap_chars - 16].rstrip() + " ...[truncated]" if len(output) > cap_chars else output


def make_replicate_bundle(
    *,
    bundle_id: str,
    case: dict[str, Any],
    model_key: str,
    original_runs: Sequence[dict[str, Any]],
    edited_runs: Sequence[dict[str, Any]],
    removed_cue_summary: str,
    strict_type_combo_value: str,
) -> dict[str, Any]:
    replicates: list[dict[str, Any]] = []
    for idx, (original, edited) in enumerate(zip(original_runs, edited_runs)):
        replicates.append(
            {
                "replicate": idx,
                "original": {
                    "answer": original.get("predicted_answer"),
                    "correct": bool(original.get("is_correct")),
                    "rationale": compact_text(original.get("rationale") or original.get("explanation") or original.get("raw_response")),
                },
                "edited": {
                    "answer": edited.get("predicted_answer"),
                    "correct": bool(edited.get("is_correct")),
                    "rationale": compact_text(edited.get("rationale") or edited.get("explanation") or edited.get("raw_response")),
                },
                "answer_changed": original.get("predicted_answer") != edited.get("predicted_answer"),
            }
        )
    return {
        "bundle_id": bundle_id,
        "case_id": case.get("case_id"),
        "dataset": case.get("dataset"),
        "example_id": case.get("example_id"),
        "model_key": model_key,
        "removed_image_hint_summary": removed_cue_summary,
        "strict_type_combo": strict_type_combo_value,
        "question": case.get("question"),
        "options": case.get("options"),
        "correct_answer": case.get("correct_answer"),
        "correct_answer_text": case.get("correct_answer_text"),
        "replicates": replicates,
    }


def build_cue_use_judge_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    return {**bundle, "decision_rules": list(CUE_USE_DECISION_RULES)}


def build_packed_cue_use_judge_payload(case: dict[str, Any], bundles: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": "Judge every model_bundles item independently for image-hint cue use and harmful cue reliance.",
        "case": case,
        "decision_rules": [
            "Judge only visible model rationales and answers.",
            "cue_use_replicates: original rationale explicitly mentions or clearly relies on the removed artificial image hint, e.g. arrow/circle/box/marker/highlight/marked region.",
            "Do not count ordinary medical image reasoning unless it references the artificial mark or marked/arrowed/circled/highlighted region.",
            "bad_use_positive_replicates: original_correct=true, edited_correct=false, and removed hint use plausibly explains the drop.",
            "A bundle can have cue use without bad-use positivity.",
            "Return lean JSON only. For each input bundle, preserve bundle_index and model_key. Keep evidence quotes very short.",
        ],
        "model_bundles": list(bundles),
    }


# ---------------------------------------------------------------------------
# Repeated-run summaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeanCI:
    mean: float
    ci_low: float
    ci_high: float
    n: int


def normal_mean_ci(values: Iterable[float], *, z: float = 1.96) -> MeanCI:
    xs = [float(value) for value in values]
    n = len(xs)
    if n == 0:
        return MeanCI(float("nan"), float("nan"), float("nan"), 0)
    mean = sum(xs) / n
    if n == 1:
        return MeanCI(mean, mean, mean, n)
    var = sum((value - mean) ** 2 for value in xs) / (n - 1)
    half = z * math.sqrt(var / n)
    return MeanCI(mean, mean - half, mean + half, n)


def summarize_metric_by_repeat(
    rows: Iterable[dict[str, Any]],
    *,
    metric_key: str,
    repeat_key: str = "replicate",
) -> MeanCI:
    by_repeat: dict[int, float] = {}
    for row in rows:
        repeat = int(row.get(repeat_key) or 0)
        by_repeat[repeat] = by_repeat.get(repeat, 0.0) + float(row.get(metric_key) or 0.0)
    return normal_mean_ci(by_repeat.values())
