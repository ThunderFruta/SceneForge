# SceneForge

SceneForge is a computer graphics project for turning 2D images into usable 3D assets and scenes.

The repository was reset on 2026-05-24. The previous Python prototype, generated outputs, sample assets, tests, and local environment artifacts were removed so the next implementation could start cleanly.

For the original project idea and first milestone, read `BEFORE_README.md`.

## Current Prototype

The current prototype takes one image plus aligned depth, proposes visible object/plane masks from depth+edge evidence, and writes detector evidence:

- `detections.json`
- `overlay.png`

Final primitive labels come from object enrichment and geometric fitting, not the detector. The active detector scaffold uses depth+edge channels and leaves primitive labels unassigned so downstream geometry can choose box/sphere/cylinder/cone/plane. The intended learned replacement has the same contract: depth+edge or depth/edge/3D evidence in, instance masks out, primitive type chosen downstream from 3D residuals/fusion. The pipeline can then fit those detections to rough 3D geometric primitives from aligned synthetic depth and writes:

Detector construction goes through `Segmentation/factory.py`. Active depth+edge detection is available without YOLO imports; YOLO segmenters are lazy-loaded only for explicit legacy/debug commands.

- `primitive_fits.json`
- `fit_overlay.png`
- `fitted_scene.blend`
- `depth_check/` diagnostics comparing input depth against fitted-scene depth

Supported primitive labels are:

- `sphere`
- `cylinder`
- `cone`
- `box`
- `plane`
- `torus`
- `tube`
- `arch`
- `unknown`

V1 uses local model paths only. It does not download weights at runtime.

Local model paths in this workspace:

- Active depth-edge scaffold does not require model weights.
- Learned detector slot: `--detector-model` routes to `Segmentation/learned_depth_edge_segmenter.py`, which loads a Primitive3D class-agnostic point-cloud instance segmentation checkpoint.
- Dataset generation writes `instance_dataset_manifest.json` for the future depth+edge/3D instance-mask model contract.
- Future instance-detector checkpoints should live under `Models/InstanceDetector/`.
- YOLO 11 medium segmentation: `Models/YOLO/yolo11m-seg.pt`
- Trained synthetic primitive YOLO segmentation: `Models/YOLO/sceneforge-primitives-yolo11m-seg.pt`
- Trained V2 synthetic primitive YOLO segmentation: `Models/YOLO/sceneforge-primitives-yolo11m-seg-v2.pt`
- RGBD YOLO26l curriculum checkpoints should be written under `Models/YOLO/` when trained.
- Edge model providers should use `Models/Edges/DexiNed/` when configured.
- Wireframe providers should use `Models/Wireframe/HAWP/` when configured.
- Mesh providers should use `Models/Mesh/TripoSR/` when configured.
- CLIP: `Models/CLIP/clip-vit-base-patch32`

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Usage

```bash
.venv/bin/python run.py detect-shapes \
  --image Assets/Fixtures/OpenVocabulary/open_vocab_smoke_objects.png \
  --depth path/to/depth.png \
  --output Output/Latest/detect
```

`detect-shapes` defaults to `--backend depth-edge`: detector labels are proposal evidence (`plane` or `unknown`), while `primitive_label` stays `unknown` until enrichment/fitting selects geometry.

Legacy/debug path using YOLO detector labels as primitive labels:

```bash
.venv/bin/python run.py detect-shapes \
  --backend rgb-yolo \
  --image Assets/Fixtures/OpenVocabulary/open_vocab_smoke_objects.png \
  --detector-weights Models/YOLO/sceneforge-primitives-yolo11m-seg-v2.pt \
  --primitive-source detector-label \
  --output Output/Latest/detect \
  --device auto
```

The RGB YOLO backend is legacy/debug. It suppresses lower-confidence duplicate detections whose boxes overlap a kept detection above `0.7` IoU. Use `--overlap-iou-threshold 0` to disable that post-processing. `--backend real` remains as a deprecated alias for old commands.

Use a trained 4-channel RGBD detector:

```bash
.venv/bin/python run.py detect-shapes \
  --backend rgbd-yolo \
  --image Assets/Fixtures/OpenVocabulary/open_vocab_smoke_objects.png \
  --depth path/to/depth.png \
  --detector-weights Models/YOLO/sceneforge-yolo26l-rgbd-stage5.pt \
  --primitive-source none \
  --output Output/Latest/detect \
  --device auto
```

Build per-object evidence packs after detection:

```bash
.venv/bin/python run.py enrich-objects \
  --image Assets/Fixtures/OpenVocabulary/open_vocab_smoke_objects.png \
  --depth path/to/depth.png \
  --detections Output/Latest/detect/detections.json \
  --edge-backend simple \
  --wireframe-backend none \
  --mesh-backend none \
  --output Output/Latest/enrich
```

`--edge-backend simple` is a lightweight classical image edge detector. Use `--edge-backend none` to omit edge evidence entirely.

Real providers are explicit and never auto-download weights. The current local model layout is:

```text
Models/Edges/DexiNed/          # OpenCV DexiNed ONNX edge model
Models/Mesh/TripoSR/           # TripoSR repo, checkpoint, and local DINO dependency
Models/Wireframe/HAWP/         # HAWP repo and hawpv*.pth checkpoints
Models/Depth/DepthAnythingV3/  # downloaded future depth checkpoint, not used in V1
```

Open-vocabulary segmentation is the primary real proposal path once the local GroundingDINO and SAM3 readiness audit passes. First create the local layout and reviewed setup script without running network operations:

```bash
.venv/bin/python run.py prepare-open-vocab-layout --root Models/OpenVocabulary
```

Then review `Models/OpenVocabulary/setup_open_vocab_sources.sh` before running it. SAM3 checkpoint access is gated by Hugging Face approval/authentication, so keep that step explicit. You can audit the full non-inference readiness state at any point:

```bash
.venv/bin/python run.py audit-open-vocab-readiness \
  --root Models/OpenVocabulary \
  --backend groundingdino-sam3 \
  --output Output/Latest/open_vocab_readiness.json
```

Preflight the expected local layout before running real inference:

```bash
.venv/bin/python run.py check-open-vocab-integration \
  --backend groundingdino-sam3 \
  --groundingdino-repo-dir Models/OpenVocabulary/GroundingDINO/repo \
  --groundingdino-config Models/OpenVocabulary/GroundingDINO/repo/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --groundingdino-checkpoint Models/OpenVocabulary/GroundingDINO/weights/groundingdino_swint_ogc.pth \
  --sam3-repo-dir Models/OpenVocabulary/SAM3/repo \
  --sam3-model-dir Models/OpenVocabulary/SAM3/hf \
  --output Output/Latest/open_vocab_preflight.json
```

When preflight passes, check that the local repo APIs import in this Python environment without loading checkpoints:

```bash
.venv/bin/python run.py probe-open-vocab-imports \
  --backend groundingdino-sam3 \
  --groundingdino-repo-dir Models/OpenVocabulary/GroundingDINO/repo \
  --sam3-repo-dir Models/OpenVocabulary/SAM3/repo \
  --output Output/Latest/open_vocab_import_probe.json
```

When both checks pass, run the guarded proposal-only smoke test against `Assets/Fixtures/OpenVocabulary/open_vocab_smoke_objects.png`:

```bash
.venv/bin/python run.py run-open-vocab-smoke \
  --root Models/OpenVocabulary \
  --backend groundingdino-sam3 \
  --output Output/Latest/open_vocab_smoke.json
```

Or run the underlying command directly:

```bash
.venv/bin/python run.py detect-shapes \
  --backend groundingdino-sam3 \
  --image Assets/Fixtures/OpenVocabulary/open_vocab_smoke_objects.png \
  --open-vocab-root Models/OpenVocabulary \
  --text-prompt-preset scene-primitives-v1 \
  --output Output/Latest/detect \
  --device auto
```

GroundingDINO and SAM3 outputs are weak proposal evidence only. SceneForge still chooses final primitive labels during enrichment and geometric fitting.

Run the primary real reconstruction proposal path after readiness passes:

```bash
.venv/bin/python run.py reconstruct-scene \
  --reference-blend Assets/Samples/shapes.blend \
  --detector-backend groundingdino-sam3 \
  --open-vocab-root Models/OpenVocabulary \
  --text-prompt-preset scene-primitives-v1 \
  --edge-backend simple \
  --wireframe-backend none \
  --mesh-backend none \
  --output Output/Latest \
  --device auto
```

This path writes `detect/proposal_quality.json` beside `detections.json` and records open-vocabulary prompt/readiness/source metadata in detection and run-status reports.

Use real providers only when those local adapters and weights exist:

```bash
.venv/bin/python run.py enrich-objects \
  --image Output/Latest/source/rgb.png \
  --depth Output/Latest/source/depth.png \
  --detections Output/Latest/detect/detections.json \
  --edge-backend dexined \
  --edge-model-dir Models/Edges/DexiNed \
  --wireframe-backend hawp \
  --wireframe-model-dir Models/Wireframe/HAWP \
  --mesh-backend triposr \
  --mesh-model-dir Models/Mesh/TripoSR \
  --output Output/Latest/enrich \
  --device 0
```

HAWP writes per-object `objects/NN/wireframe_crop.png` overlays and `objects/NN/wireframe.json` line/junction evidence. The real adapter shells out to the local HAWP repo and never downloads weights. Use `--wireframe-backend none` to omit wireframe evidence. Use `--mesh-backend none` to skip mesh candidates.

Run object-level TripoSR reconstruction after detection/completion has written `Output/Latest/objects`:

```bash
.venv/bin/python run.py reconstruct-objects \
  --objects Output/Latest/objects \
  --backend triposr \
  --model-dir Models/Mesh/TripoSR \
  --device auto \
  --source completed
```

This writes `triposr_mesh.obj` and `triposr_metadata.json` in each processed object folder plus `Output/Latest/objects/triposr_manifest.json`.

Render one combined evidence overlay from an existing detection/enrichment run:

```bash
.venv/bin/python run.py render-evidence-overlay \
  --image Output/Latest/source/rgb.png \
  --detections Output/Latest/detect/detections.json \
  --enrichment Output/Latest/enrich/object_enrichment.json \
  --output Output/Latest/enrich/evidence_overlay.png
```

The combined overlay draws detector masks/boxes, dense edge evidence, wireframe lines when available, and mesh-candidate status markers. `reconstruct-scene` writes this automatically at `Output/Latest/enrich/evidence_overlay.png`.

Fit detected primitive masks to a Blender scene with synthetic depth:

```bash
.venv/bin/python run.py fit-primitives \
  --image Assets/Fixtures/OpenVocabulary/open_vocab_smoke_objects.png \
  --depth path/to/depth.png \
  --detections Output/Latest/detect/detections.json \
  --enrichment Output/Latest/enrich/object_enrichment.json \
  --output Output/Latest/fit \
  --fov-degrees 70 \
  --reference-blend path/to/original.blend
```

Run the full geometry-first reconstruction path. This renders deterministic RGB/depth source truth from the active camera in `Assets/Samples/shapes.blend`, proposes masks with the depth-edge detector scaffold, enriches objects, fits primitives, and writes `Output/Latest/fitted_scene.blend`:

```bash
.venv/bin/python run.py reconstruct-scene \
  --reference-blend Assets/Samples/shapes.blend \
  --edge-backend simple \
  --wireframe-backend none \
  --mesh-backend none \
  --output Output/Latest \
  --device auto
```

`--device auto` selects CUDA when PyTorch reports a CUDA GPU and otherwise falls back to CPU. Use `--device cpu`, `--device cuda`, or `--device 0` to force a specific inference device. `reconstruct-scene` preflights selected real providers before touching `Output/Latest`. Missing real provider paths fail immediately. Public commands do not expose fake detector, mesh, or wireframe providers; tests keep deterministic test doubles under `Tests/Fakes/`.

Depth maps must be aligned grayscale images where white is close and black is far. By default, `--fov-degrees` is interpreted as horizontal FOV to match Blender's horizontal sensor fit. The fitting stage exports only geometric primitives: `sphere`, `cylinder`, `cone`, `box`, and `plane`; unknown detections become box proxies.
Use `--camera-shift-x` and `--camera-shift-y` for small Blender camera framing tweaks when a fit is globally offset but the object scale is close.
Every fitted run renders the generated scene's depth back out and writes `depth_check/depth_check.json`, `depth_check/depth_check_side_by_side.png`, and depth difference heatmaps.
The fit report also records the selected fit mode, primitive label source, and per-object depth error metrics. Objects with high depth mismatch are marked `needs_review`.
The fitting stage writes one `.blend` file: `fitted_scene.blend`. For the standard `Output/Latest/fit` run layout, that scene is saved at `Output/Latest/fitted_scene.blend` so it is easy to open. The default final layout is `--final-layout camera`, which preserves the same SceneForge camera-space axes used by source rendering, depth fitting, and depth validation: X right, Y depth away from camera, Z up. Use `--final-layout ground` only when you want a repositioned upright inspection scene.
When `--reference-blend` is provided with `--final-layout ground`, the final scene maps the original `.blend` active camera pose and framing onto the generated inspection layout.

Render metric view sets from any `.blend` file:

```bash
blender -b Output/Latest/fitted_scene.blend \
  --python Tools/Scripts/render_metrics_views.py -- \
  --output Output/Latest/metrics/generated
```

Metric view sets include the actual active-camera preview, orthographic RGB, grayscale depth, world-normal RGB, and isometric RGB views.

Compare original/generated metric view sets and include source-camera depth diagnostics:

```bash
.venv/bin/python run.py compare-metrics \
  --original-metrics Output/Latest/metrics/original \
  --generated-metrics Output/Latest/metrics/generated \
  --depth-check Output/Latest/fit/depth_check/depth_check.json \
  --output Output/Latest/metrics
```

This writes `Output/Latest/metrics/summary.json`, `comparison/metrics_comparison.csv`, and side-by-side comparison images.

## Development

```bash
.venv/bin/python -m pytest
```

Generated outputs and the local virtual environment are ignored by git. Keep the active run under `Output/Latest/`. Move older runs under timestamped folders in `Output/Archive/`, such as `Output/Archive/20260524_153000/`.

## Synthetic Primitive Dataset / Legacy YOLO Training

Generate a primitive dataset with Blender headless. The first-class training artifact is `instance_dataset_manifest.json`; YOLO labels are still generated as compatibility output for legacy experiments:

```bash
blender -b --python Tools/Dataset/generate_primitives_dataset.py -- \
  --output Datasets/PrimitiveShapes \
  --count 100 \
  --fov-degrees 70 \
  --seed 20260524 \
  --width 640 \
  --height 640 \
  --min-shapes 1 \
  --max-shapes 3 \
  --write-instance-masks
```

Train the segmentation-first Primitive3D detector from that manifest:

```bash
.venv/bin/python run.py train-instance-detector \
  --manifest Datasets/PrimitiveShapesRGBDTarget/no_plane_hard/instance_dataset_manifest.json \
  --config Configs/InstanceDetector/primitive_3d_segmentation.json \
  --output Models/InstanceDetector/no_plane_primitive3d \
  --device auto
```

This writes `primitive_3d_segmenter.pt`, `training_summary.json`, and `eval_summary.json`. The learned backend is class-agnostic: it outputs instance masks only, then enrichment/fitting chooses primitive labels.

This writes ignored local training data under `Datasets/PrimitiveShapes/`:

- `train/images/`
- `train/masks/`
- `train/labels/`
- `val/images/`
- `val/masks/`
- `val/labels/`
- `test/images/`
- `test/masks/`
- `test/labels/`
- `data.yaml`

The generator still writes quick projected labels, but the better training path is to rebuild labels from the rendered visible instance masks:

```bash
.venv/bin/python Tools/Dataset/masks_to_yolo_labels.py \
  --dataset Datasets/PrimitiveShapes
```

Render labeled preview images from the YOLO labels:

```bash
.venv/bin/python Tools/Dataset/render_label_previews.py \
  --dataset Datasets/PrimitiveShapes
```

Create a thermal-style RGB preview from a grayscale depth map:

```bash
.venv/bin/python Tools/Scripts/colorize_depth_map.py \
  --input Output/Latest/render/shapes_depth_from_blend.png \
  --output Output/Latest/render/shapes_depth_thermal.png
```

This writes:

- `train/annotations/`
- `val/annotations/`
- `test/annotations/`

Generate a staged RGBD curriculum dataset with Blender:

```bash
.venv/bin/python run.py generate-rgbd-dataset \
  --curriculum-stage 1 \
  --images-per-class 100 \
  --output Datasets/PrimitiveShapesRGBD \
  --width 640 \
  --height 640 \
  --render-samples 8 \
  --seed 20260524
```

Stages 1-5 train the base classes (`sphere`, `box`, `cylinder`, `cone`, `plane`). Stages 6-7 add `torus`, `tube`, and `arch`. Stages 8-10 are scene-composition stages that still use the existing primitive labels: furnishings/common combinations, minimally occluded scenes, and dense low-poly scenes. Each stage uses a split-first layout:

- `train/images/`, `train/rgb/`, `train/rgbd/`, `train/depth/`, `train/labels/`, `train/masks/`, `train/annotations/`
- `val/images/`, `val/rgb/`, `val/rgbd/`, `val/depth/`, `val/labels/`, `val/masks/`, `val/annotations/`
- `test/images/`, `test/rgb/`, `test/rgbd/`, `test/depth/`, `test/labels/`, `test/masks/`, `test/annotations/`
- `data_rgbd.yaml`, `stage_manifest.json`, and `validation_report.json`

RGBD training images are RGBA PNGs where RGB stores the render and alpha stores aligned normalized depth. The default split is 70% train, 20% validation, and 10% test; `data_rgbd.yaml` includes `train: train/images`, `val: val/images`, `test: test/images`, and `channels: 4`.
Curriculum stages use larger primitive scale ranges and filter labels by minimum projected screen area so tiny objects and near-invisible slivers do not become training targets. Clean stages also reject placements with high projected overlap, so Stage 2 and Stage 3 stay mostly unobstructed after projection. For manual generator runs, use `--shape-scale-min`, `--shape-scale-max`, `--min-screen-area-ratio`, `--max-screen-overlap-ratio`, and `--render-samples` to override those defaults. Instance masks are rendered with a single color-ID pass per scene and then split into one mask per object.

Large dataset generation can be parallelized with Blender workers. By default, `--shards auto` chooses an initial worker count from CPU count and dataset size, then runs disjoint index chunks and adjusts the next wave of workers about every five minutes based on observed throughput. Manual shard counts still use static modulo sharding.

```bash
.venv/bin/python run.py generate-rgbd-dataset \
  --curriculum-stage 2 \
  --count 1000 \
  --output Datasets/PrimitiveShapesRGBD \
  --width 640 \
  --height 640 \
  --render-samples 8 \
  --shards auto \
  --seed 20260526
```

Rendering uses Blender EEVEE; it can use GPU rendering when Blender has a working GPU backend, while placement, mask splitting, label conversion, and PNG writing run mostly on CPU.
For adaptive mode tuning, set `SCENEFORGE_ADAPTIVE_CHUNK_SIZE` or `SCENEFORGE_ADAPTIVE_ADJUST_SECONDS` before running the command.

Generate a target-style adaptation dataset from the real `shapes.blend` layout:

```bash
.venv/bin/python run.py generate-target-rgbd-dataset \
  --reference-blend Assets/Samples/shapes.blend \
  --output Datasets/PrimitiveShapesRGBDTarget/shapes_blend \
  --count 500 \
  --width 640 \
  --height 640 \
  --render-samples 8 \
  --shards auto \
  --seed 20260526
```

Generate the held-out exact target evaluation image. This writes all samples to `test/` and should not be used for training:

```bash
.venv/bin/python run.py generate-target-rgbd-dataset \
  --reference-blend Assets/Samples/shapes.blend \
  --output Datasets/TargetEval/shapes_blend \
  --count 1 \
  --eval-only \
  --width 640 \
  --height 640 \
  --render-samples 16 \
  --seed 20260525
```

Evaluate a trained RGBD checkpoint on that target eval:

```bash
.venv/bin/python run.py eval-rgbd-yolo \
  --data Datasets/TargetEval/shapes_blend/data_rgbd.yaml \
  --weights Models/YOLO/sceneforge-yolo26l-rgbd-stage5-target.pt \
  --output Output/Latest/eval_target_shapes \
  --split test \
  --imgsz 640 \
  --batch 8 \
  --device 0
```

Once the stage 1-5 datasets, target adaptation dataset, and target eval set exist, run the staged training and final target eval:

```bash
Tools/Scripts/train_rgbd_curriculum_to_target.sh
```

If a generation run is interrupted, rerun it with the same stage, count, and output plus `--finish`. Existing complete samples are skipped and only missing indices are rendered:

```bash
.venv/bin/python run.py generate-rgbd-dataset \
  --curriculum-stage 2 \
  --count 1000 \
  --output Datasets/PrimitiveShapesRGBD \
  --width 640 \
  --height 640 \
  --render-samples 8 \
  --shards auto \
  --finish \
  --seed 20260526
```

Legacy comparison training for the RGBD YOLO26l config:

```bash
.venv/bin/python run.py train-rgbd-yolo \
  --data Datasets/PrimitiveShapesRGBD/stage1_single_clean/data_rgbd.yaml \
  --model Configs/YOLO/yolo26l_seg_rgbd.yaml \
  --output Models/YOLO/sceneforge-yolo26l-rgbd-stage1.pt \
  --epochs 200 \
  --imgsz 640 \
  --batch auto \
  --device cuda
```
