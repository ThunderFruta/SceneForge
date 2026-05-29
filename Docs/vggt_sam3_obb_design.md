# VGGT + SAM3 + OBB Design (No explicit primitive classifier in v1)

## Objective

Use `SAM3` masks as the object proposal source, use `VGGT` for geometric substrate, and fit each object with an oriented 3D bounding box (OBB) first. Use `Hunyuan3D` output as a detail mesh that is aligned to the fitted box, not as the primary geometry source.

This design explicitly replaces the current “primitive-type-first” assumption for this path with an OBB-first placement strategy.

## Why this path now

- `SAM3` gives strong object proposals, but masks still need quality gates for occlusion, merged objects, shadows, transparency, and rectangular fallback cases.
- `VGGT` gives coherent geometry and depth cues, but not always production-quality meshes.
- Fitting an OBB to VGGT-derived object points is stable, deterministic, and avoids hard class mistakes (e.g., sphere vs cylinder).
- Alignment of a detail mesh to an OBB is straightforward and easier to validate than direct mesh-to-scene registration.

## Proposed end-to-end flow

1. **Proposal stage (existing)**
   - Run `groundingdino-sam3` or `sam3` detector backend.
   - Persist `detections.json` with masks and proposal IDs.

2. **Geometry capture (VGGT)**
   - Run `VGGT` on the original image and obtain per-pixel depth/point support
     plus usable camera parameters.
   - Normalize outputs into SceneForge camera space: X right, Y depth away from
     camera, Z up.
   - Prefer lightweight artifacts for this path:
     - `objects_vggt/vggt_depth.png` (or depth variant used in run)
     - `objects_vggt/vggt_points.xyz` (or run-local equivalent)
     - `objects_vggt/vggt_camera.json`

3. **Object piece extraction**
   - For each `detection_id`, use its mask to sample VGGT depth/points.
   - Crop a per-object point cloud with metadata: point count, valid-depth
     ratio, bbox bounds, mask quality, and failure or review reason.

4. **OBB fitting first**
   - Fit oriented bounding box parameters from each per-object cloud:
     - center `(x,y,z)`
     - size `(sx,sy,sz)`
     - orientation matrix / quaternion
     - quality score (fit residual + support ratio)
   - Record if fallback used (e.g., degenerate PCA, too few points).

5. **Detail mesh alignment (optional)**
   - Run `Hunyuan3D` object reconstruction per object.
   - Normalize detail mesh local frame.
   - Align detail mesh transform to the object OBB (rigid or rigid+ICP refine).
   - Validate against OBB depth/silhouette projection.

6. **Scene assembly**
   - Export final scene assets with both:
     - box/OBB placement geometry for inspection
     - optional aligned Hunyuan detail meshes
   - Keep OBB/contact placement as authoritative in v1.

## Contracts and schema extensions

- Keep current proposal outputs as the stable input contract:
  - `detections.json` from SAM3/groundingdino-sam3
  - `overlay.png`
- Add dedicated object geometry output at `objects_vggt/object_geometry.json`:
  - `detection_id`
  - `fit_mode: vggt_mask_crop`
  - `obb_center`, `obb_extent`, `obb_rotation`, `obb_quality`
  - `point_count`, `coverage_ratio`, `mask_quality`, `failure_reason`, and
    `needs_review`
  - `source`: includes `vggt_depth_source`, `sam3_mask_id`, and
    `reconstruction_backend`
- Write snap/plane decisions to `scene_alignment.json`; do not write placement
  through retired proxy reports or compatibility aliases.

## Execution flow in CLI

Recommended CLI pattern:

```bash
python3 run.py detect-shapes \
  --image Assets/Samples/Chairs.jpg \
  --backend groundingdino-sam3 \
  --open-vocab-root Models/OpenVocabulary \
  --text-prompt-preset scene-primitives-v1 \
  --output Output/Latest/detect

python3 run.py reconstruct-objects \
  --objects Output/Latest/objects \
  --backend hunyuan3d

python3 run.py fit-object-placements \
  --image Assets/Samples/Chairs.jpg \
  --detections Output/Latest/detect/detections.json \
  --objects Output/Latest/objects \
  --background-planes Output/Latest/background/planes.json \
  --geometry-backend vggt \
  --output Output/Latest/alignment
```

`fit-object-placements` owns original-image VGGT object sampling, OBB/contact
estimation, optional detail-mesh alignment, and `scene_alignment.json` output.

## Minimal risk containment

- Keep depth+edge and primitive-proxy paths out of active execution unless a
  future plan explicitly reactivates them.
- Keep OBB placement outputs separate from removed primitive-proxy flags.
- Write clear quality metadata so weak masks, rectangular fallback masks, or
  empty VGGT crops do not silently pass.

## Failure handling

If per-object VGGT geometry is insufficient:
- mark `obb_fit_status = failed`
- set `failure_reason` (e.g., `too_few_points`, `invalid_depth`, `degenerate_covariance`)
- keep object in output with no mesh placement
- continue reconstruction for remaining objects

For OBB quality confidence below threshold:
- flag the object as `needs_review`
- leave the object unplaced or mark it for review; do not route through any retired fallback

## Evaluation checklist

- Per object:
  - OBB fit inlier ratio above threshold
  - point coverage above threshold
  - mesh alignment depth residual below threshold (if detail mesh enabled)
- Run-level:
  - number of placed objects vs detection count
  - failed object count by reason
  - total scene placement runtime

## Open questions

- Single-image VGGT sufficiency: gate OBB placement behind configurable
  confidence thresholds and mark weak crops `needs_review`.
- OBB orientation convention: use SceneForge camera-space transforms with X
  right, Y depth away from camera, and Z up.
- Viewer artifacts may include both box-only placement geometry and aligned
  detail meshes, but `objects_vggt/object_geometry.json` remains the placement
  authority.
- Emit both an object-geometry overlay and a mesh-snap/alignment overlay when
  composed-scene artifacts are available.
