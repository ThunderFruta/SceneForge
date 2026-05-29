# Current Changes

This file tracks notable project changes while SceneForge is still small.

## 2026-05-29

- Added `Docs/empty_room_vggt_background_design.md` to define the OpenAI empty-room inpaint, dual-VGGT background/object geometry, plane extraction, and object-to-plane snapping plan.
- Added `Docs/plane_detection_design.md` to define the additive geometry-first path for large structural planes, including subtype metadata, optional `plane_detections.json`, future CLI flags, quality policy, and fit-contract compatibility.

## 2026-05-27

- Added `ObjectReconstruction/` with `run.py reconstruct-objects`, a direct TripoSR object-crop-to-mesh stage that defaults to `Output/Latest/objects/*/completed_crop.png`, writes `triposr_input.png`, `triposr_mask.png`, `triposr_mesh.obj`, per-object metadata, and `triposr_manifest.json`.
- Switched default object reconstruction to GPU-backed Hunyuan3D with TripoSR retained as a fallback backend. Hunyuan3D consumes completed object crops and writes `hunyuan3d_mesh.obj`, `hunyuan3d_mesh.glb`, per-object metadata, and `hunyuan3d_manifest.json`.
- Added the first open-source integration scaffold for local GroundingDINO and SAM3 repos as proposal-only segmentation backends.
- Added `sam3` and `groundingdino-sam3` detector backends with lazy imports, local repo/checkpoint path flags, text prompts, and backend metadata that keeps primitive labels downstream.
- Added `Docs/integration_contract.md` to pin the adapter contract, model directory convention, and no-runtime-download rule for open-source integrations.
- Added `run.py check-open-vocab-integration` and `Tools/Integration/open_vocab_preflight.py` to validate local GroundingDINO/SAM3 repo and checkpoint layout before real inference.
- Added `run.py probe-open-vocab-imports` and `Tools/Integration/open_vocab_import_probe.py` to isolate external repo import/API issues before checkpoint loading or inference.
- Added `run.py prepare-open-vocab-layout` and `Tools/Integration/open_vocab_setup.py` to create the local open-vocabulary model layout plus a reviewed setup script before network cloning/downloading.
- Added `run.py audit-open-vocab-readiness` and `Tools/Integration/open_vocab_readiness.py` to combine setup, path preflight, import probe, next steps, and smoke-test command reporting.
- Added deterministic open-vocabulary smoke-test assets under `Assets/Fixtures/OpenVocabulary/` and updated generated smoke-test commands to use them.
- Added `run.py run-open-vocab-smoke` and `Tools/Integration/open_vocab_smoke.py` to guard and run the first GroundingDINO/SAM3 `detect-shapes` smoke test after readiness passes.
- Hardened the GroundingDINO/SAM3 setup path with local `.venv` CUDA/NVCC handling, a reproducible GroundingDINO PyTorch API patch, and explicit SAM3 Hugging Face auth/cache readiness reporting.
- Promoted GroundingDINO/SAM3 toward the primary real reconstruction proposal path with `--open-vocab-root`, `scene-primitives-v1` prompt presets, readiness-gated reconstruction, open-vocabulary run metadata, and proposal-quality reporting.
- Added guided no-argument mode for public SceneForge entrypoints: `run.py` now opens an interactive DINO/SAM-first workflow wizard, integration/tool scripts print equivalent commands before running defaults, and Blender helper scripts can build the required `blender --background ... --python ... -- ...` command without the user knowing the separator syntax.

## 2026-05-24

- Reset the repository for a fresh implementation direction.
- Deleted the previous Python prototype source tree, tests, configuration placeholders, tools, sample assets, generated outputs, local caches, and local virtual environment.
- Recreated `README.md` as a minimal reset-state overview.
- Updated `structure.md` so it no longer describes the deleted prototype as current code.
- Added the first fresh prototype: object segmentation plus primitive shape labeling from a single 2D image.
- Added `run.py detect-shapes` with real local YOLO/CLIP backends and deterministic fake backends for tests.
- Added JSON report and overlay writers.
- Added tests for image loading, primitive label validation, report serialization, fake pipeline output, no-detection output, and CLI failure/success paths.
- Added `requirements.txt` and created `.venv` for local development.
- Added `Tools/Dataset/generate_primitives_dataset.py` to create Blender-rendered primitive-shape YOLO segmentation datasets.
- Generated a local ignored 100-image dataset under `Datasets/PrimitiveShapes/` with 80 train images, 20 validation images, and matching YOLO segmentation label files.
- Added `Tools/Dataset/render_label_previews.py` and generated labeled preview images under `Datasets/PrimitiveShapes/labeled_images/`.
- Trained `Models/YOLO/sceneforge-primitives-yolo11m-seg.pt` from `yolo11m-seg.pt` on the 100-image synthetic primitive dataset.
- Added `--primitive-source detector-label` so the CLI can use trained primitive detector labels directly without CLIP.
- Added duplicate-overlap suppression for real YOLO detections to reduce lower-confidence class duplicates on the same object.
- Added rendered instance mask output to the Blender dataset generator plus `Tools/Dataset/masks_to_yolo_labels.py` so future training can use visible object silhouettes instead of projected convex hull labels.
- Generated `Datasets/PrimitiveShapesV2/` with 1000 images and mask-derived labels, then trained `Models/YOLO/sceneforge-primitives-yolo11m-seg-v2.pt`.
- Added `run.py fit-primitives` to fit detected masks and aligned grayscale synthetic depth maps to rough 3D geometric primitives, writing `primitive_fits.json`, `fit_overlay.png`, and `fitted_scene.blend`.
- Added `PrimitiveFitting/` with pinhole unprojection, mask point-cloud sampling, simple primitive fitters, geometric-only output, and Blender export.
- Changed the default fitting and synthetic dataset camera FOV to 70 degrees.
- Added optional fitted-scene Blender camera shift controls for small 2D framing adjustments.
- Changed primitive fitting to use horizontal FOV by default so unprojection matches the Blender camera sensor fit.
- Replaced the ad hoc `Output/` contents with `Output/Latest/` for the active run and `Output/Archive/<timestamp>/` for archived runs.
- Added thermal-style RGB depth preview generation for easier visual inspection of grayscale depth maps.
- Added automatic fitted-scene depth validation outputs so generated geometry can be compared against the source depth map on every fit run.
- Added metrics view rendering for original/generated `.blend` scenes: isometric, axis orthographic, axis depth, and axis normal outputs.
- Added active-camera preview rendering to metrics so each metric set includes the same camera view the `.blend` opens with.
- Added hybrid primitive fit candidate selection so camera-space fitting can choose depth/PCA candidates instead of always using a front-facing silhouette.
- Added per-object depth validation metrics, primitive label source metadata, and `needs_review` quality flags for high-mismatch fits.
- Added `run.py compare-metrics` to compare original/generated metric view folders and write `Output/Latest/metrics/summary.json`.
- Changed `fitted_scene.blend` to be the upright ground-style inspection scene; camera-space depth validation now uses a temporary internal `.blend`.
- For `Output/Latest/fit` runs, moved the single saved `.blend` up to `Output/Latest/fitted_scene.blend` for easier access.
- Added `fit-primitives --reference-blend` so the final scene can reuse an original `.blend` active camera framing.
- Started the RGBD detector curriculum path: added stage presets, RGB/depth/RGBA dataset outputs, validation reports, YOLO26l 4-channel configs, RGBD YOLO detection wiring, and a `train-rgbd-yolo` CLI entrypoint.
- Updated RGBD curriculum datasets to use 70/20/10 train/val/test splits and a split-first layout such as `train/images`, `train/labels`, `train/depth`, and `train/rgbd`; RGBA training images continue to store normalized depth in alpha.
- Added per-split `annotations/` preview images that draw YOLO polygons, bounding boxes, and indexed class labels for dataset auditing.
- Added adaptive dataset generation with `--shards auto`, which starts near an estimated sweet spot and adjusts worker count between disjoint index chunks about every five minutes.
- Added `--primitive-source none` so detector outputs can keep YOLO labels as weak evidence while leaving final primitive labels unassigned.
- Added `ObjectEnrichment/`, `EdgeDetection/`, and `MeshReconstruction/` with CPU-safe fake providers, deterministic object evidence packs, an explicit geometry-classifier authority module, geometry scores, edge overlays, and advisory mesh candidate paths.
- Added `run.py enrich-objects` and fake-provider `run.py reconstruct-scene` orchestration with `Output/Latest/run_status.json` and archive/replace output lifecycle.
- Added `fit-primitives --enrichment` so primitive fitting uses geometry-selected labels and records detector-label, edge, wireframe, and mesh audit fields in `primitive_fits.json`.
- Downloaded local open-source model assets for DexiNed, TripoSR, HAWP, and Depth Anything V3 SMALL under `Models/`.
- Wired the DexiNed real edge provider through the downloaded OpenCV ONNX model.
- Wired the TripoSR real mesh provider through the downloaded TripoSR checkpoint plus local DINO dependency, with a CPU marching-cubes fallback for environments where `torchmcubes` cannot build.
- Added `run.py generate-target-rgbd-dataset` for rendering labeled RGBD training/adaptation data directly from `Assets/Samples/shapes.blend`.
- Added `run.py eval-rgbd-yolo` for measuring a trained RGBD checkpoint on train/val/test splits, including the held-out `shapes.blend` target eval.
- Earlier YOLO-mask-to-evidence pipeline contract: `detect-shapes` defaulted to unassigned primitive labels, `reconstruct-scene` used RGBD YOLO masks, real edge/mesh providers were lazy-imported only when selected, and reconstruct runs wrote a compact `metrics/summary.json`. This has since been superseded by the depth+edge default scaffold.
- Wired HAWP into enrichment through a new `WireframeDetection/` provider boundary. `enrich-objects` and `reconstruct-scene` now support `--wireframe-backend none|hawp`, write per-object `wireframe_crop.png` and `wireframe.json` when enabled, and keep the real HAWP adapter lazy/import-safe.
- Added `SceneGeometry/coordinate_contract.py` so source renders, detection reports, enrichment crops, fit reports, exported `.blend` files, and metrics views share a single camera-space contract: horizontal FOV, X right, Y depth away from camera, Z up, and white-close depth. Final fitted `.blend` export now defaults to exact camera-space layout; the older ground inspection layout is opt-in with `--final-layout ground`.
- Added `run.py render-evidence-overlay` to compose detector masks/boxes, dense edge maps, wireframe JSON lines, and mesh-candidate status markers into one audit image. `reconstruct-scene` writes `enrich/evidence_overlay.png` automatically after enrichment.
- Cleaned fake provider plumbing out of public runtime commands. At that stage, `reconstruct-scene` required RGBD YOLO detection; it now defaults to depth+edge detection. Edge providers are `none|simple|dexined`, mesh providers are `none|triposr`, and wireframe providers are `none|hawp`. Deterministic fakes now live under `Tests/Fakes/` for tests only.

## 2026-05-25

- Promoted fused reconstruction to the fitting execution contract: fitting now consumes `fused_state` (YOLO, depth, edge, wireframe, mesh evidence) as the primary primitive label source.
- Hardened fused report loading to normalize missing modality buckets and per-label score coverage.
- Updated enrichment/fitting tests and CLI wording so fused evidence is treated as the reconstruction contract input, not geometry/legacy label fallbacks.
- Added a one-pass primitive depth refiner after initial fitting: it renders camera-space fitted depth, compares source/generated residuals per object, ignores weak/sparse overlap evidence, applies bounded center-depth and size corrections, accepts the refined state only if rendered depth scoring does not regress, then exports the final `.blend` from the accepted primitive state.
- Expanded fitted-depth diagnostics with foreground IoU plus missing/extra foreground ratios so refinement acceptance accounts for silhouette mismatch, not only depth-value residuals.
- Added the same foreground IoU and missing/extra foreground ratios to per-object fit quality, with low object IoU marked as `needs_review`.
- Added `fit-primitives --no-depth-refinement` so direct fitting runs can compare initial-only geometry against the accepted refined state without editing code.
- Added a final `fit_quality_summary` verdict and `quality_gate_passed` flag to `primitive_fits.json` so each run explicitly reports whether the accepted geometry is `good`, `usable_needs_review`, or `needs_review` from depth and foreground metrics.
- Made `depth_refinement.json` explicit even when refinement is disabled, preventing stale refinement diagnostics from being confused with the current output.
- Added `Tools/Scripts/check_fit_quality.py` to make `primitive_fits.json` quality gates scriptable after a generated `.blend` run, including a required fitted `.blend` existence check and support for passing either a report path or a run directory.
- Added `fit-primitives --require-quality-gate` so command-line fitting can fail fast unless the final accepted geometry passes `primitive_fits.json` quality checks.
- Added `reconstruct-scene --require-quality-gate` so full scene reconstruction can fail before metrics rendering unless the final primitive fit quality gate passes, including resumed runs.
- Enabled Ultralytics `retina_masks` for RGB and RGBD YOLO prediction to preserve higher-resolution mask boundaries in detection reports.
- Added configurable legacy `reconstruct-scene --detector-confidence` and `--detector-overlap-iou-threshold` for RGBD YOLO runs, defaulting reconstruction confidence to `0.20` so rotated/occluded primitives just below the previous `0.25` cutoff could survive into fusion review.
- Added experimental RGBD YOLO input channel weighting in B,G,R,D order; the default remains equal `0.25,0.25,0.25,0.25` because inference-time depth-heavy weighting broke detections for the current equal-channel-trained checkpoint.
- Added `generate-target-rgbd-dataset --object-rotation-degrees` to render target-blend training samples with deterministic random per-object Euler rotation jitter while preserving exact-first samples.
- Added `generate-target-rgbd-dataset --random-object-rotation` to use fully random per-object Euler rotations within the configured rotation range instead of jittering around the original pose.
- Promoted `Models/YOLO/sceneforge-yolo26l-rgbd-target-rot2000.pt` as the default `reconstruct-scene` RGBD detector checkpoint after training on 2000 random-rotation target samples.
- Preserved fitted primitive rotation in `original-camera` Blender exports by mapping camera-space rotation matrices through the reference camera transform instead of exporting upright world-axis proxies.
- Added a 2D mask-axis rotation fallback for camera-silhouette cylinder/cone fits so visible image orientation can propagate into the fitted 3D primitive when depth PCA is rejected.
- Tuned fusion for weak detector labels: high-confidence YOLO still anchors, low-confidence true positives can survive, but clearly stronger depth geometry can override weak detector mislabels such as rotated spheres predicted as cylinders.
- Added a depth-label margin before weak detector override so ambiguous depth geometry does not erase low-confidence YOLO true positives such as rotated cones.
- Added `Tools/Scripts/compare_fit_quality.py` to compare baseline and candidate fit reports, require both fitted `.blend` deliverables by default, and optionally exit nonzero when the candidate does not improve depth score.
- Hardened fit quality scripts so missing `fit_quality_summary` reports fail explicitly instead of being treated as weak metric failures.
- Added `--json` output to fit quality gate scripts so reconstruction quality and A/B comparison results are machine-readable without parsing human text.

## 2026-05-26

- Added a separate no-plane generalization track with `generate-no-plane-rgbd-dataset`, detector recall reporting in `test_blends.py`, and promotion gating through `Tools/Scripts/check_generalization_summary.py`.
- Added `generate-plane-context-rgbd-dataset` for balanced foreground primitives with labeled room-like floor/wall `plane` context, plus `Tools/Scripts/train_plane_context_generalization.sh` for fine-tuning after the no-plane detector is stable.
- Split blend-test detector coverage into foreground object recall and plane recall so large walls/floors do not hide missed foreground objects.
- Added `--device auto` inference support for YOLO/reconstruction/evidence commands so CUDA is selected when available and CPU is used as fallback; detection reports now record requested and resolved detector devices.
- Started dismantling YOLO as the primary reconstruction dependency. `detect-shapes`, `reconstruct-scene`, `generate.py`, and `test_blends.py` now default to a depth+edge geometry-first segmentation scaffold; RGB/RGBD YOLO remains available as legacy/debug/training comparison backends.
- Added `Segmentation/depth_edge_segmenter.py` and a backend-info contract so future trained depth-edge instance models can replace the deterministic scaffold without changing reconstruction orchestration.
- Added backend-neutral `Runtime/device.py` device resolution so CUDA/CPU selection is no longer owned by the YOLO segmenter.
- Added `LearnedSegmentationModelSpec`, `Segmentation/learned_depth_edge_segmenter.py`, and `--detector-model` routing so the next detector can be a depth+edge/3D instance-mask model while primitive classification stays in geometry/fusion.
- Moved YOLO segmenter/training imports behind legacy code paths and added `Tools/Dataset/instance_manifest.py`; dataset generators now write detector-neutral instance-mask manifests before legacy YOLO label conversion.
- Added `Segmentation/factory.py` so `run.py` asks for detector runtimes through a backend-neutral factory instead of directly constructing YOLO or depth-edge segmenters in command handlers.
- Added `Configs/InstanceDetector/depth_edge_3d_instance.json` and `Tools/Training/instance_detector.py` plus `train-instance-detector` and `eval-instance-detector` scaffold commands that consume `instance_dataset_manifest.json`; RGBD YOLO training remains a legacy comparison path.
- Added the first Primitive3D segmentation-first detector implementation: `primitive-3d-segmenter` loads a PointNet-style class-agnostic point embedding checkpoint from `--detector-model`, projects clustered 3D point instances back into `detections.json`, and keeps primitive labels downstream. `train-instance-detector` and `eval-instance-detector` now build RGB/depth/camera point-cloud caches, train/evaluate the local PyTorch model, and write `primitive_3d_segmenter.pt`, `training_summary.json`, and `eval_summary.json`.
- Added `run.py generate-perfect-detections` / `generateperfect` plus `Tools/Scripts/render_perfect_blend_masks.py` to render authoritative visible masks and primitive labels directly from labeled `.blend` geometry for detector debugging.

## Current State

SceneForge currently has a Python CLI prototype that detects visible object/plane proposals from depth+edge evidence and writes `detections.json` plus `overlay.png`. Detector labels are weak proposal evidence only; `detect-shapes` leaves `primitive_label` unassigned by default, `enrich-objects` writes geometry/fusion-selected primitive labels from mask/depth/edge evidence, then `fit-primitives --enrichment` fits rough geometric 3D proxies and exports a Blender scene.

The active fallback detector scaffold uses depth and edge; the learned `primitive-3d-segmenter` path uses RGB/depth/camera-derived point features and outputs class-agnostic instance masks. Local YOLO segmentation weights, CLIP, and detector-label primitive assignment are legacy/debug paths, not the default authority for primitive labels. Dataset generation writes `instance_dataset_manifest.json` as the detector-neutral training contract, while YOLO labels remain compatibility artifacts. The code does not download model weights at runtime. Tests use isolated test doubles and do not require external model files.

Use `BEFORE_README.md` for the original project idea, then update these docs when the new direction becomes more concrete.

## 2026-05-23

- Added `AGENTS.md` with guidance for coding agents.
- Added `structure.md` to describe the intended repository layout.
- Added `current_changes.md` to track early changes.
- Added `project_preferences.md` to capture naming and project conventions.
- Updated `structure.md` with an approved modular tree inspired by HCRBot's subsystem layout.
- Created the initial modular directory tree on disk.
- Added `Docs/architecture.md` and `Docs/tree.md`.
- Added the first Python CLI prototype for image/depth to textured OBJ export.
- Added tiny PPM/PGM fixtures under `Assets/Fixtures/`.
- Added tests for image loading, depth loading, mesh generation, UV generation, OBJ export, and the pipeline.
- Added `pyproject.toml` with Pillow and pytest configuration.
- Added `.gitignore` for Python caches and generated mesh outputs.
- Added sample PNG inputs in `Assets/Samples/` and generated the first local Blender-importable OBJ bundle under `Output/`.
- Changed the CLI to write `.blend` by default through Blender background import.
- Added `--obj` to keep a sidecar OBJ bundle only when requested.
- Moved the extracted room model into `Assets/Samples/Room/` as a local ignored sample asset.
- Converted the local room OBJ sample into `Assets/Samples/Room/room.blend`.
- Updated the edited room blend with random per-object colors for easier inspection.
- Rendered `Assets/Samples/Room/room.blend` to `Assets/Samples/Room/room_render.png`.
- Re-rendered `room_render.png` with an interior-facing camera view.
- Generated `Assets/Samples/Room/room_render_depth.png` from the same room camera view.
- Generated `Output/room_reconstructed.blend` from `room_render.png` plus `room_render_depth.png`.
- Added `--mode relief|structured` with relief as the default.
- Added structured scene mode with connected depth-region analysis, plane parts, and detail relief patches.
- Generated `Output/room_structured.blend` from the room render/depth pair.
- Cleared generated files from `Output/`.
- Changed CLI output handling to create organized timestamped run folders.
- Updated structured mode to ignore near-black invalid depth cells and project parts into camera-space instead of pure image-card coordinates.
- Added `Geometry/Planes/plane_fitter.py`: PCA plane fit via a pure-Python 3×3 Jacobi eigenvalue solver.
- Rewrote plane part construction: cells are unprojected to a point cloud, a plane is fitted, and each corner is placed by ray-plane intersection so surface orientation (wall/floor/ceiling) comes from actual depth data rather than average depth.
- Changed structured mode to output fitted planes by default and hide leftover detail relief patches unless `--details` is provided.
- Added masked plane mesh generation so structured plane regions preserve their connected-cell silhouettes instead of becoming bounding-box rectangles.
- Added region cleanup for small one-row/one-column plane fragments.
- Added automatic `preview.png` rendering beside every generated `.blend` output.
- Changed generated `.blend` files to save a source-facing active camera so `preview.png` and Blender camera view compare against the input image instead of an arbitrary inspection angle.
- Corrected structured camera-space vertical orientation after Blender OBJ import.
- Updated Blender OBJ import to preserve SceneForge axes, mirror the imported scan into the source-facing view, and scale generated `.blend` scenes up 4x for easier inspection.
- Changed imported texture materials to emission materials so previews are not dominated by Blender lighting/shadow artifacts.
- Added a canonical projection module for image/depth to 3D mapping: X right, Y depth away from camera, Z up.
- Removed exporter-level mesh mirroring and normal flipping; generated geometry now faces the right way before Blender import.
- Added deterministic per-vertex normal generation for relief meshes and structured scene parts.
- Updated OBJ export to write `vn` records and `v/vt/vn` face references when normals are present.
- Added structured-mode scan solidification with thin side walls on visible plane/detail boundaries.
- Added `--solidify`, `--no-solidify`, `--solidify-thickness`, and `--depth-edge-threshold` CLI controls for structured mode.
- Added depth-edge thresholding for structured plane/detail mesh faces before solidification.
- Tightened structured depth bucket grouping and plane-size promotion so small fragments are less likely to become coarse occluding plane chunks.
- Added `coverage_000`, a behind-plane valid-depth relief fallback in structured `--details` output, to fill visible floor/wall/object areas missed by plane segmentation.
- Increased the CLI default resolution to `128` for cleaner structured scan masks.
- Generated `Output/20260523_175613_structured_room_solidified/room_solidified.blend` from the room render/depth pair with `--details --obj`.
- Generated `Output/20260523_182651_structured_room_coverage_128/room_coverage_128.blend` from the room render/depth pair with `--details --obj`.
- Added top-level `Segmentation/` with shared labels, mask data, manual RGB mask loading, heuristic segmentation, mask-to-region conversion, and SAM 3 provider documentation.
- Added `--segmentation none|mask|auto` and `--mask PATH` for structured mode.
- Added `Assets/Samples/Room/room_render_mask.png` as the first manual segmentation baseline for the room sample.
- Added `Docs/segmentation.md` and `Configs/Segmentation/README.md`.
- Updated `AGENTS.md` with explicit subagent collaboration guidance for parallel work.
- Generated `Output/20260523_194731_structured_room_masked/room_masked.blend` from the room render/depth/mask set with `--details --obj`.
- Tests: 63 passed.
- Added `Tools/Scripts/view_blend.py` to render multi-view previews, optional orbit snapshots, and `.glb` from any `.blend` file for quick inspection.
- Added JSON report output (`*_view_report.json`) with mesh counts, bounds, and per-view camera transforms for agent-friendly diagnostics.
- Added `Geometry/Cleanup/` as a visible cleanup subsystem for structured mode.
- Added structured mask cleanup for small same-label holes and tiny non-border islands.
- Added structured mesh cleanup for small internal boundary-loop caps, obvious horn/spike face rejection, and large occlusion gap diagnostics.
- Added `--cleanup`, `--no-cleanup`, `--hole-fill-size`, and `--spike-threshold` CLI controls.
- Added `cleanup_counts` and occlusion gap fields to structured `metrics.json`.
- Tests: 72 passed.
- Added `Geometry/DepthValidity/` to separate invalid/no-data depth from valid near-black far-depth surfaces.
- Added `--depth-invalid-mode black|threshold|none` and `--min-valid-depth` CLI controls for structured mode.
- Changed structured mode's default invalid-depth policy from threshold-like near-black removal to exact-black invalid handling, preserving stable dark back walls by default.
- Added `depth_validity_counts` to structured `metrics.json`.
- Tests: 83 passed.

## Previous State

Before the 2026-05-24 reset, SceneForge had a first Python CLI MVP that wrote `.blend` and `preview.png` output from an image and optional depth map when Blender was installed. Sidecar OBJ output was optional via `--obj` and included explicit normals. Structured mode included masked plane output, depth validity handling, cleanup, solidification, and optional segmentation guidance. That implementation has been removed and should be treated as historical context only.
