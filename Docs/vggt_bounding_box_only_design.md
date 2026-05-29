# VGGT Bounding Box Only Design

This document defines a simpler near-term SceneForge path that adds VGGT as the
geometry source and uses per-object bounding boxes for placement, without empty
room plane detection or object-to-plane snapping.

The goal is to get an inspectable scene-placement stage working before adding
background reconstruction complexity. This path should produce stable object
geometry reports from the original image only.

## Objective

Use SAM3 or GroundingDINO-SAM3 masks as object proposals, run VGGT on the
original object-filled image, sample VGGT geometry inside each proposal mask,
and fit each object to a 3D bounding box.

The bounding box is the V1 placement primitive. Hunyuan3D or TripoSR meshes may
be aligned into the box for visual detail, but the mesh is not the placement
authority.

## Non-Goals

This path intentionally does not include:

- OpenAI empty-room generation;
- VGGT on an empty-room image;
- background plane extraction;
- floor, wall, ceiling, road, or concrete-floor subtype detection;
- snapping objects to background planes;
- retired primitive-proxy fitting or compatibility outputs.

Those steps can remain in separate design tracks. This path is the smaller
"get VGGT into SceneForge" milestone.

## Pipeline

1. Run object proposals.

   Use the existing `sam3` or `groundingdino-sam3` proposal path to write
   `detections.json`, `overlay.png`, and per-object mask workspaces. Detector
   labels stay weak evidence only.

2. Run VGGT on the original image.

   Use the original object-filled frame as the VGGT input. Normalize VGGT depth,
   points, and camera data into the SceneForge camera-space contract:

   - `X` points image right;
   - `Y` points away from the source camera along depth;
   - `Z` points image up.

3. Crop object geometry from proposal masks.

   For each detection, sample the VGGT point map inside the object mask. Record
   point count, valid-point ratio, mask quality, bounding rectangle, and any
   crop failure reason.

4. Fit a 3D bounding box.

   Fit either an axis-aligned bounding box (AABB) or oriented bounding box (OBB)
   from the masked VGGT points.

   Recommended V1 behavior:

   - prefer OBB when point support is stable;
   - fall back to AABB when OBB orientation is degenerate;
   - fail the object cleanly when there are too few valid VGGT points;
   - mark weak masks, rectangular fallback masks, sparse crops, and degenerate
     boxes as `needs_review`.

5. Optionally align detail meshes.

   If Hunyuan3D or TripoSR meshes exist for an object, normalize the mesh local
   frame and place it inside the fitted box. The alignment can start as simple
   center-and-scale placement. Later ICP or silhouette checks can refine this,
   but V1 should stay inspectable.

6. Export review artifacts.

   Write JSON reports and optional visual/debug artifacts that show each
   detected object, its VGGT crop, and its fitted 3D bounding box.

## Report Contract

Recommended object geometry report:

```text
objects_vggt/bounding_boxes.json
```

Recommended top-level fields:

- `schema_version`
- `image_path`
- `detections_path`
- `vggt_depth_path`
- `vggt_points_path`
- `vggt_camera_path`
- `coordinate_contract`
- `boxes`
- `model_info`

Recommended per-object fields:

- `detection_id`
- `detector_label`
- `detector_confidence`
- `mask_path`
- `mask_quality`
- `bbox_xyxy`
- `box_type`
- `center_xyz`
- `extent_xyz`
- `rotation_matrix`
- `point_count`
- `valid_point_ratio`
- `coverage_ratio`
- `fit_residual`
- `needs_review`
- `failure_reason`
- `detail_mesh_path`
- `detail_mesh_transform`

`box_type` should be one of:

- `obb`
- `aabb`
- `failed`

`rotation_matrix` may be identity for AABB fits.

## CLI Shape

Suggested staged command:

```bash
python3 run.py fit-vggt-boxes \
  --image Input/Image/example.png \
  --detections Output/Latest/detect/detections.json \
  --objects Output/Latest/objects \
  --geometry-backend vggt \
  --output Output/Latest/objects_vggt
```

Optional flags:

- `--box-mode obb|aabb|auto`
- `--min-valid-points`
- `--min-valid-point-ratio`
- `--include-detail-meshes`
- `--device auto`

The command should not require `background/planes.json` or any empty-room
artifacts.

## Quality Policy

Do not silently place weak geometry. A detection should be preserved in the
report even when box fitting fails, with `box_type=failed`, `needs_review=true`,
and a clear `failure_reason`.

Suggested failure reasons:

- `missing_mask`
- `rectangular_fallback_mask`
- `too_few_points`
- `invalid_vggt_depth`
- `degenerate_covariance`
- `box_extent_out_of_range`
- `detail_mesh_missing`
- `detail_mesh_alignment_failed`

Rectangular fallback masks may be allowed only when an explicit override is
provided; otherwise they should remain review-required.

## Acceptance Checks

The first implementation should be considered useful when it can:

- consume existing proposal outputs without changing `detections.json`;
- run VGGT on the original image and write normalized geometry artifacts;
- produce one bounding-box record per detection;
- fail individual objects without failing the whole run;
- export a simple review view with boxes and optional detail meshes;
- keep all plane detection and snapping artifacts absent.

This path should make it easy to inspect whether VGGT-derived object placement is
good enough before investing in empty-room background geometry.
