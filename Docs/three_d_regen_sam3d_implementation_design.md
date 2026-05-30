# 3D-RE-GEN and SAM3D Implementation Design

## Purpose

This document converts the useful parts of 3D-RE-GEN and SAM3D into a concrete SceneForge implementation plan.

The goal is not to vendor either project wholesale. SceneForge should adopt the practical contracts:

- Use SAM/GroundingDINO masks as stable proposal and fitting evidence.
- Generate an empty-room image with explicit mask control.
- Run VGGT on original and empty-room images to recover shared camera-space geometry.
- Fit support planes from empty-room geometry.
- Place object meshes with constrained support-plane optimization.
- Treat SAM 3D Objects as an optional object-mesh backend, not as the scene-layout authority.

## Source Lessons

### 3D-RE-GEN

3D-RE-GEN is most useful as a system design reference. Its strongest ideas are:

- Compositional scene reconstruction instead of one monolithic mesh.
- Context-aware object completion before 2D-to-3D reconstruction.
- Empty-room generation as a first-class background lane.
- VGGT or Dust3R point-cloud recovery for camera and geometry evidence.
- 4-DoF floor/support-plane fitting instead of unconstrained 7-DoF object placement.
- Rendered silhouette and 3D point evidence as explicit fitting losses.
- Background geometry as a physical constraint, not only a visual mesh.

SceneForge should copy these ideas, not its runtime stack. The upstream repo assumes a heavy environment with PyTorch3D, Open3D, pycolmap, submodules, and tightly coupled scripts. SceneForge should keep small, replaceable modules and stable JSON reports.

### SAM 3D Objects

SAM 3D Objects is useful as an object-level reconstruction backend. It can consume an image and mask and return object geometry, texture, and pose/layout estimates.

SceneForge should use it as:

- A future backend under `ObjectReconstruction/`.
- A comparison target beside Hunyuan3D and TripoSR.
- A mesh generator for difficult objects where Hunyuan3D fails.

SceneForge should not use it as:

- The empty-room generator.
- The authoritative background geometry source.
- The final placement solver.
- A default dependency until license and hardware requirements are acceptable.

### Pointcept SegmentAnything3D

The older Pointcept SAM3D project is primarily a 3D segmentation system. Its useful idea is projecting 2D SAM masks into 3D point evidence and merging regions.

For SceneForge, this maps to:

- Object-mask-to-VGGT-point extraction.
- Mask erosion before projection to avoid boundary depth contamination.
- Per-object 3D region reports.

It should not become the main SceneForge architecture.

## Proposed SceneForge Pipeline

### Stage 1: Object proposals

Input:

- Source RGB image.
- Text prompts or default open-vocabulary labels.

Output:

- `detections.json`
- `overlay.png`
- Per-object `mask.png`
- Per-object `crop.png`
- Per-object metadata with label, box, confidence, mask area, and prompt source.

Implementation notes:

- Continue using the detector-neutral proposal report.
- Keep primitive labels unassigned.
- Preserve masks at source resolution.
- Store enough mask provenance to distinguish SAM, GroundingDINO-SAM, manual, and future SAM3 masks.

### Stage 2: Empty-room edit input

Input:

- Source RGB image.
- Selected object masks.
- Optional structural-protection labels: `floor`, `wall`, `ceiling`, `window`, `door`.

Output:

- `foreground_removal_mask.png`
- `empty_room_openai_mask.png`
- `empty_room_edit_input.png`
- `empty_room_metadata.json`

Implementation notes:

- Combine selected masks into a full-frame removal mask.
- Dilate object masks enough to remove contact shadows and edge halos.
- Avoid erasing protected structural masks when structural labels exist.
- Record mask dilation radius, selected object ids, protected object ids, and final mask coverage.
- Keep a deterministic neutral-fill fallback so tests do not require image-edit API calls.

### Stage 3: Empty-room generation

Input:

- `empty_room_edit_input.png`
- `empty_room_openai_mask.png`
- Source image for visual context.

Output:

- `empty_room.png`
- `empty_room_metadata.json`

Prompt policy:

```text
Remove the selected foreground objects and furniture.
Preserve the same camera framing, perspective, room layout, walls, floor, ceiling, lighting, and image resolution.
Do not invent new furniture or decorations.
Fill occluded floor and wall regions consistently with the visible room.
```

Implementation notes:

- Prefer explicit image-edit masking over prompt-only removal.
- Add metadata fields for backend, prompt, seed if available, image dimensions, mask path, and fallback mode.
- Fail clearly if the backend returns a resized or reframed output unless the caller allows resizing.

### Stage 4: Optional Application-Querying object completion

Purpose:

Use 3D-RE-GEN-style context-aware object completion for occluded objects.

Input:

- Source RGB image.
- Object mask and crop.
- Object label.

Output:

- `application_query.png`
- `completed_crop.png`
- `completed_mask.png`
- `object_completion_metadata.json`

Layout:

- Left panel: full source image with the target object outlined.
- Right panel: square card named `Extracted Object`.
- The image model fills the right card with an amodal, object-only product view.

Prompt policy:

```text
The left panel shows the source scene with the target object marked.
Replace the right panel with a complete isolated render of only the marked object.
Use the source scene for perspective, material, color, and lighting cues.
Do not include floor, walls, platforms, shadows, or other objects.
Keep the object uncropped and centered on a transparent or white background.
```

Implementation notes:

- Keep this optional because it adds another generative step.
- Use it only when masks indicate truncation, occlusion, or low completion confidence.
- Regenerate `completed_mask.png` from the completed output, not from the original visible mask.

### Stage 5: VGGT original and empty-room geometry

Input:

- Source RGB image.
- `empty_room.png`.

Output:

- `objects_vggt/depth.png`
- `objects_vggt/point_map.npy` or existing point-map artifact.
- `objects_vggt/camera.json`
- `empty_room_vggt/depth.png`
- `empty_room_vggt/point_map.npy` or existing point-map artifact.
- `empty_room_vggt/camera.json`
- Optional merged/aligned reconstruction report.

Implementation notes:

- Prefer running original and empty-room images through the same VGGT invocation when practical, so both outputs share camera assumptions.
- Preserve SceneForge camera coordinates: X right, Y depth away from camera, Z up.
- Write explicit transform metadata for any VGGT/OpenCV-to-SceneForge conversion.
- Avoid hidden scale normalization. If scale alignment is applied, write it into JSON.

### Stage 6: Empty-room plane fitting

Input:

- Empty-room VGGT point map.
- Empty-room image.
- Optional structural masks.

Output:

- `plane_detections.json`
- `empty_room_planes.glb`
- `plane_debug_overlay.png`

Required plane fields:

- `plane_id`
- `label`
- `normal`
- `point`
- `basis_u`
- `basis_v`
- `extent`
- `bounds_2d`
- `inlier_count`
- `residual_mean`
- `residual_p95`
- `confidence`
- `source`

Implementation notes:

- Fit large structural planes first: floor, back wall, side walls.
- Use robust RANSAC or deterministic sampled fitting followed by SVD refinement.
- Regularize labels toward expected room axes only after the raw fit is measured.
- Texture planes from `empty_room.png` using source-camera projection.
- Keep low-confidence planes in the report but mark them as `needs_review`.

### Stage 7: Object VGGT target extraction

Input:

- Original-image VGGT point map.
- Per-object masks.
- Per-object boxes.

Output:

- `object_geometry.json`
- Per-object `visible_points.ply` or compact sampled point artifact.
- Per-object valid-point debug mask.

Implementation notes:

- Erode masks before point extraction to avoid mixed foreground/background edges.
- Reject points with invalid depth/confidence.
- Compute object AABB/OBB in SceneForge camera space.
- Record visible point count and coverage ratio.
- Do not infer full object volume from visible points alone. Treat them as placement evidence.

### Stage 8: Object mesh reconstruction

Input:

- `completed_crop.png` or object crop.
- `completed_mask.png` or object mask.

Output:

- `hunyuan3d_textured.glb`, `triposr_mesh.obj`, or future `sam3d_objects.glb`.
- `object_reconstruction_metadata.json`.

Backend policy:

- `hunyuan3d`: default practical object mesh backend.
- `triposr`: fallback or comparison backend.
- `sam3d-objects`: future optional backend after licensing and hardware checks.

Implementation notes:

- Normalize output orientation and units before placement.
- Preserve original backend mesh for audit.
- Write a processed mesh path separately if cleanup/remeshing/support-sheet removal is applied.

## Placement and Fitting Design

### Support selection

Input:

- Object detections.
- Object VGGT geometry.
- Empty-room planes.
- Existing placed support objects.

Output:

- `placement/object_supports.json`

Support modes:

- `floor_plane`
- `tabletop_plane`
- `shelf_plane`
- `counter_plane`
- `unknown_support_5dof`
- `floating_review`

Selection rules:

- Large furniture defaults to the floor unless evidence says otherwise.
- Small objects whose 2D bottom overlaps a table/counter candidate should test that support first.
- If no support is reliable, use `unknown_support_5dof` and mark the object for review.
- Never hardcode label-specific poses. Labels can bias candidate order, not override evidence.

### 4-DoF support-plane transform

Use this for reliable support planes.

Parameters:

- `u`: translation along support basis U.
- `v`: translation along support basis V.
- `yaw`: rotation around support normal.
- `scale`: uniform scale.

Constraints:

- Object support contact stays on the plane.
- No free translation along plane normal.
- No pitch or roll unless the support itself is tilted.
- Object bottom/contact footprint must remain within support bounds, with configurable tolerance.

This is the central 3D-RE-GEN idea SceneForge should adopt.

### 5-DoF fallback transform

Use this when support is unknown or weak.

Parameters:

- `x`
- `y`
- `z`
- `yaw`
- `scale`

Constraints:

- No pitch/roll.
- Penalize deviation from likely support depth.
- Penalize floating and penetration relative to nearby planes.
- Mark final status as review unless metrics are strong.

### Initialization

Before optimization:

- Center mesh at object VGGT target centroid or support-projected target.
- Estimate starting scale from 2D projected box size and visible VGGT extents.
- Estimate yaw with a small grid search over projected silhouette or 3D OBB alignment.
- Place the mesh contact footprint on the selected support plane.

Initialization report fields:

- `initial_center`
- `initial_scale`
- `initial_yaw`
- `initial_support_contact_point`
- `initialization_method`
- `candidate_count`

### Losses

#### Silhouette loss

Render the candidate mesh from the source camera and compare to the object mask.

Metrics:

- Mask IoU.
- Dice loss.
- Missing foreground ratio.
- Extra foreground ratio.
- Projected bbox L1/L2 loss.

Use the mask as the main placement signal for image-space alignment.

#### VGGT point proximity loss

Compare sampled mesh surface points to the visible object VGGT points.

Metrics:

- Point-to-mesh distance mean.
- Point-to-mesh distance p95.
- Mesh-to-point sampled distance mean.
- Visible centroid error.

Use this as a depth/contact signal, not as full-shape truth.

#### Support contact loss

Ensure the object support footprint touches the selected plane.

Metrics:

- Contact height error.
- Penetration depth.
- Floating gap.
- Support footprint coverage.

For 4-DoF mode this should be mostly enforced by parameterization, with diagnostics still reported.

#### Background bounds loss

Penalize objects that cross structural bounds in empty-room geometry.

Checks:

- Behind back wall.
- Through side wall.
- Below floor.
- Far outside room bounds.

This should be a soft loss during fitting and a hard diagnostic in final quality.

#### Object overlap loss

After individual placement, compute object-object intersection or approximate bbox overlap.

Metrics:

- Overlap volume estimate.
- Support hierarchy violations.
- 2D projection overlap changes.

This can start as a post-fit diagnostic before becoming a joint optimization term.

### Optimization strategy

V1 should not require PyTorch3D.

Recommended first implementation:

- Use coarse grid search for yaw and scale.
- Use deterministic local search over support-plane translation.
- Render simple projected bounding/silhouette proxies in software.
- Score candidates with weighted losses.
- Write full candidate and final score reports.

V2 can add differentiable rendering:

- PyTorch3D or nvdiffrast behind an optional backend.
- Same JSON contracts.
- Same loss names and report fields.

This keeps tests lightweight and avoids making heavy ML/render dependencies mandatory.

## Placement Reports

### `placement/object_fit_targets.json`

Fields:

- `object_id`
- `label`
- `mask_path`
- `visible_points_path`
- `source_box`
- `mask_area`
- `vggt_point_count`
- `target_centroid`
- `target_aabb`
- `target_obb`
- `support_candidates`

### `placement/object_placements.json`

Fields:

- `object_id`
- `support_id`
- `support_mode`
- `transform_model`
- `translation`
- `rotation`
- `scale`
- `support_contact_point`
- `support_normal`
- `losses`
- `diagnostics`
- `status`

Statuses:

- `accepted`
- `usable_needs_review`
- `needs_review`
- `failed`

### `placement/placement_quality.json`

Fields:

- `summary_status`
- `object_count`
- `accepted_count`
- `review_count`
- `failed_count`
- `loss_summary`
- `support_mode_counts`
- `penetration_warnings`
- `floating_warnings`
- `overlap_warnings`
- `background_bounds_warnings`
- `review_items`

## Acceptance Criteria

An object placement is `accepted` when:

- Mask IoU is above the configured threshold.
- Projected bbox error is below the configured threshold.
- VGGT point proximity is finite and not an outlier.
- Support penetration and floating gap are below tolerance.
- The support footprint is plausible for the selected support.
- Object does not cross high-confidence room bounds.

An object is `usable_needs_review` when:

- Image-space fit is plausible but support evidence is weak.
- VGGT points are sparse or noisy.
- The object is heavily occluded.
- The mesh backend produced unusual scale or support geometry.

An object is `needs_review` when:

- It floats, sinks, or visibly misses its mask.
- It intersects strong background planes.
- It overlaps another placed object in an implausible way.
- Support selection falls back to unknown 5-DoF.

## Implementation Milestones

### Milestone 1: Contract-only placement reports

- Define report dataclasses or typed dictionaries.
- Emit object support candidates from detections and planes.
- Emit object fit targets from masks and VGGT points.
- No mesh optimization yet.

### Milestone 2: Deterministic 4-DoF candidate search

- Implement support-plane coordinate helpers.
- Generate candidate `u`, `v`, `yaw`, and `scale` values.
- Score projected bbox and simple silhouette proxy.
- Write `object_placements.json`.

### Milestone 3: Mesh-aware scoring

- Sample mesh vertices or surface points.
- Add point proximity losses.
- Add support footprint and bounds checks.
- Add object overlap diagnostics.

### Milestone 4: Empty-room/Application-Querying improvements

- Add optional Application-Querying layout for object completion.
- Add strict empty-room prompt and metadata contract.
- Add output framing/resolution checks.

### Milestone 5: Optional SAM 3D Objects backend

- Add lazy adapter with local-only model/checkpoint paths.
- Consume existing object crop/mask artifacts.
- Write backend-neutral object reconstruction metadata.
- Compare against Hunyuan3D and TripoSR outputs.

## Non-Goals

- Do not vendor the 3D-RE-GEN repo.
- Do not require PyTorch3D for lightweight tests.
- Do not make SAM 3D Objects the default mesh backend yet.
- Do not reconnect retired primitive-proxy commands.
- Do not solve full global scene optimization before per-object placement reports are inspectable.

## Recommended Next Build Step

Implement Milestone 1 and Milestone 2 first:

1. `choose-object-supports`
2. `build-object-fit-targets`
3. `fit-object-placements`

Keep these commands deterministic and JSON-first. A visually imperfect but inspectable 4-DoF placement lane is more useful right now than a heavy differentiable optimizer that is hard to debug.
