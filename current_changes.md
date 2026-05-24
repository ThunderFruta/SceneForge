# Current Changes

This file tracks notable project changes while SceneForge is still small.

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

## Current State

SceneForge currently has a first Python CLI MVP that writes `.blend` and `preview.png` output from an image and optional depth map when Blender is installed. Sidecar OBJ output is optional via `--obj` and now includes explicit normals. Structured mode starts with masked plane output, preserves near-black far-depth surfaces unless threshold mode is requested, filters large depth jumps, cleans small mask/mesh defects, rejects obvious spikes, and adds side-wall thickness by default. Optional segmentation masks can guide wall/floor/ceiling plane regions and object/detail relief regions. Add `--details` when you want uncertain relief patches plus the coverage fallback for better visual completeness.

## Next Likely Change

Pivot structured reconstruction toward primitive candidate completion: preserve valid far-depth scaffold surfaces first, then add room/object priors only where true occlusion gaps remain. Future SAM-style segmentation should provide segment proposals and boundaries, not trusted semantic classification.
