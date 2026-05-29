# Open Source Integration Contract

SceneForge integrates open-source vision repos as replaceable proposal providers.
The downstream contract stays stable: detector backends write `detections.json`
and object workspaces, then later stages use those proposals for mask-quality
gating, empty-room mask construction, original-image VGGT object placement, and
mesh snapping. Detector labels remain weak evidence; they do not decide geometry
or placement.

## First target: GroundingDINO + SAM3

- `sam3` uses `facebookresearch/sam3` as an open-vocabulary mask proposal backend.
- `groundingdino-sam3` uses `IDEA-Research/GroundingDINO` for text-conditioned boxes and `facebookresearch/sam3` for mask refinement when the local SAM3 API exposes box prompts.
- Both backends are proposal-only. They must not mark primitive labels as authoritative.
- Repo code and model files are local paths supplied by CLI flags; SceneForge must not download weights at runtime.

## Required CLI inputs

For SAM3-only detection:

```bash
python3 run.py detect-shapes \
  --backend sam3 \
  --image Input/Image/example.png \
  --sam3-repo-dir Models/OpenVocabulary/SAM3/repo \
  --sam3-model-dir Models/OpenVocabulary/SAM3/hf \
  --text-prompt "chair . table . box . sphere . cylinder . cone . plane ." \
  --output Output/Latest/detect
```

For GroundingDINO plus SAM3:

```bash
python3 run.py detect-shapes \
  --backend groundingdino-sam3 \
  --image Input/Image/example.png \
  --groundingdino-repo-dir Models/OpenVocabulary/GroundingDINO/repo \
  --groundingdino-config Models/OpenVocabulary/GroundingDINO/repo/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --groundingdino-checkpoint Models/OpenVocabulary/GroundingDINO/weights/groundingdino_swint_ogc.pth \
  --sam3-repo-dir Models/OpenVocabulary/SAM3/repo \
  --sam3-model-dir Models/OpenVocabulary/SAM3/hf \
  --text-prompt "chair . table . box . sphere . cylinder . cone . plane ." \
  --output Output/Latest/detect
```

## Prepare local layout

Create the expected local folder layout and a reviewed setup script without
running network operations:

```bash
python3 run.py prepare-open-vocab-layout --root Models/OpenVocabulary
```

This writes `Models/OpenVocabulary/open_vocab_setup_manifest.json` and
`Models/OpenVocabulary/setup_open_vocab_sources.sh`. Review the script before
running it because it clones repos, installs editable packages, and downloads the
public GroundingDINO checkpoint. SAM3 checkpoint access remains gated by Hugging
Face approval and authentication.

## Combined readiness audit

Use the combined audit when you want one report for setup, path preflight, import
probe, next steps, and the first smoke-test command:

```bash
python3 run.py audit-open-vocab-readiness \
  --root Models/OpenVocabulary \
  --backend groundingdino-sam3 \
  --output Output/Latest/open_vocab_readiness.json
```

This still does not load checkpoints or run inference. The generated first smoke-test command uses `Assets/Fixtures/OpenVocabulary/open_vocab_smoke_objects.png`.

## Preflight before inference

Use the repo-local preflight command before importing either external repo:

```bash
python3 run.py check-open-vocab-integration \
  --backend groundingdino-sam3 \
  --output Output/Latest/open_vocab_preflight.json
```

This checks local paths and expected repo files only. It does not import model code,
load checkpoints, download assets, or run inference.

After path preflight passes, probe imports without loading checkpoints:

```bash
python3 run.py probe-open-vocab-imports \
  --backend groundingdino-sam3 \
  --output Output/Latest/open_vocab_import_probe.json
```

This isolates Python/package issues from model-weight and inference issues.

## Guarded smoke test

After readiness reports `ready_for_smoke_test`, run the guarded smoke command:

```bash
python3 run.py run-open-vocab-smoke \
  --root Models/OpenVocabulary \
  --backend groundingdino-sam3 \
  --output Output/Latest/open_vocab_smoke.json
```

The command reruns readiness first, then executes the generated `detect-shapes`
command against `Assets/Fixtures/OpenVocabulary/open_vocab_smoke_objects.png`.

## Adapter rules

- Keep external imports lazy inside adapter execution paths. `import run` must not import GroundingDINO or SAM3.
- Convert all outputs to `Segmentation.types.SegmentDetection`.
- Use `detector_label` and `detector_confidence` as weak semantic evidence only.
- Preserve proposal-only semantics in `model_info`, using `primitive_label_policy=placement_geometry_downstream` for the active VGGT placement direction.
- Prefer explicit reduced-quality masks over hidden failure if SAM3 box-prompt support changes. Rectangular fallback masks must be marked with `mask_quality=rectangular_fallback`; downstream empty-room, VGGT crop, and placement stages should treat them as review-required unless explicitly overridden.

## Model directory convention

```text
Models/
  OpenVocabulary/
    GroundingDINO/
      repo/
      weights/
    SAM3/
      repo/
      hf/
```

The `hf/` directory is a local Hugging Face cache or checkpoint directory. Keep it
out of git with the rest of model artifacts.

## Primary proposal path

After readiness passes, the preferred active integration path is proposal detection:

```bash
python3 run.py detect-shapes \
  --backend groundingdino-sam3 \
  --image path/to/image.png \
  --open-vocab-root Models/OpenVocabulary \
  --text-prompt-preset scene-primitives-v1 \
  --output Output/Latest/detect \
  --device auto
```

`--open-vocab-root` expands to the GroundingDINO repo/config/checkpoint and SAM3 repo/cache paths. Proposal runs must fail before output mutation when the readiness audit is not `ready_for_smoke_test`. Detections remain proposal-only: GroundingDINO/SAM3 labels are detector evidence, not primitive-label authority.

Open-vocabulary detection writes proposal quality metadata in `detections.json` and `proposal_quality.json`, including object count, mask quality counts, rectangle fallback count, duplicate overlap count, tiny/empty masks, labels seen, prompt preset, prompt text, thresholds, SAM mask mode, readiness status, and local source paths. Downstream stages must read the mask-quality fields before using a proposal for empty-room removal, VGGT point cropping, or object placement.
