# Plane Detection Design

This document defines the V1 plane-first path for large structural scene
surfaces such as walls, floors, ceilings, roads, concrete slabs, and other
dominant planar regions.

The goal is to add broad surface coverage to the empty-room VGGT background
lane without reviving the retired primitive-proxy fitting path. Plane detection
should produce placement-ready plane evidence from the empty-room VGGT
background geometry first. Semantic subclass labels are metadata and guidance
for snapping object meshes, not a retired proxy contract.

## Problem

Object proposal detectors are a poor primary source for large structural
surfaces. A wall, floor, ceiling, or road can span most of the frame, be
partially occluded by foreground objects, and have weak visual texture. These
surfaces need a geometry-first pass over the empty-room VGGT background model
instead of object-level proposal masks or the object-filled original frame.

SceneForge needs this path to:

- recover large support surfaces that object segmentation may miss;
- keep foreground object detection independent from structural surface fitting;
- preserve `detections.json` as the object proposal contract while writing background plane evidence separately;
- expose quality fields so bad or ambiguous planes are reviewable.

## Scope

V1 detects connected planar regions from the empty-room VGGT background
geometry. The preferred source is the VGGT point cloud or background mesh
generated from `background/empty_room.png`, after SAM3 foreground masks have
been removed and inpainted. RGB may refine subclass labels, but RGB must not be
required for geometry acceptance.

Default plane subtypes:

- `wall`
- `floor`
- `ceiling`
- `road`
- `concrete_floor`
- `plane_unknown`

When evidence is weak, the system should fall back to a plain `plane` primitive
with `plane_subtype=plane_unknown` or no subtype metadata.

Out of scope for V1:

- heavy semantic segmentation models;
- room layout optimization;
- multi-view plane merging;
- replacing object proposal generation;
- detecting planes directly from raw SAM3 object masks;
- using the object-filled original-image VGGT pass as the primary plane source;
- changing SceneForge's camera coordinate contract.

## Coordinate Contract

Plane detection uses the existing SceneForge camera-space contract from
`SceneGeometry/coordinate_contract.py`:

- `X` points image right;
- `Y` points away from the source camera along depth;
- `Z` points image up;
- background VGGT depth/points are aligned with `background/empty_room.png`;
- depth uses the current white-close, black-far depth convention when exported
  as an image;
- units match the SceneForge camera-space contract and future Blender export path.

Plane reports must remain in the shared SceneForge camera-space frame so object
placement, snapping, and export can consume them without routeing through the
retired proxy pipeline.

## Pipeline

The proposed sequence is:

0. Receive empty-room VGGT geometry.

   Plane detection starts after the empty-room background lane has produced
   `background/empty_room.png` and VGGT-derived geometry artifacts such as a
   point cloud, depth map, camera file, or background mesh. It must not use raw
   SAM3 masks as plane candidates.

1. Normalize inputs.

   Build a valid-geometry mask from the empty-room VGGT point cloud, mesh, or
   aligned depth image. Optionally add RGB confidence weighting from
   `background/empty_room.png` when available and stable, but do not require RGB
   for candidate extraction.

2. Extract a point cloud.

   If the source is a depth map, unproject valid pixels through the shared
   SceneGeometry camera contract. If the source is already a VGGT point cloud or mesh,
   normalize it into the same SceneForge camera-space frame so candidate
   candidate generation and placement share the same coordinates.

3. Generate plane candidates.

   Use iterative RANSAC, mesh-face clustering, or point-cloud patch clustering
   with normal support. Each accepted candidate must record the plane equation,
   normal, inlier mask or mesh-face set, residuals, and support count.

4. Clean connected regions.

   Split candidate masks into connected components. Remove tiny regions, fill
   small holes, and reject thin strips that are likely edge artifacts. Keep
   occlusion boundaries; a wall behind objects should remain one plane candidate
   when the connected support is strong enough.

5. Tag the subtype.

   Apply geometry-derived priors first:

   - normal direction;
   - camera-space height;
   - vertical extent;
   - orientation relative to the image and horizon;
   - support ratio in the lower, middle, or upper frame;
   - adjacency and support relationships with other planes.

   Then apply lightweight appearance refinement only when RGB is available:

   - `road` may use lower-frame support, broad horizontal extent, outdoor-like
     color/texture cues, and weak vertical enclosure evidence;
   - `concrete_floor` may use floor-like orientation plus slab-like color and
     low-frequency texture cues;
   - ambiguous cases remain `plane_unknown`.

6. Emit placement-ready plane objects.

   Plane candidates should be written to background plane reports with optional
   `plane_subtype` and quality metadata for object snapping and review.

## Classification Heuristics

The first implementation should be intentionally simple and inspectable.

Suggested geometry-first subtype rules:

- `floor`: normal mostly points upward in camera space, support appears in the
  lower frame, and the surface is below object centroids or acts as object
  support.
- `ceiling`: normal mostly points downward or opposite floor support, support
  appears in the upper frame, and height is above most other geometry.
- `wall`: normal is mostly horizontal, vertical extent is high, and support
  reaches middle or upper frame regions.
- `road`: floor-like orientation, lower-frame dominance, long horizontal extent,
  and weak indoor wall/ceiling enclosure evidence. RGB refinement can strengthen
  but should not force this label.
- `concrete_floor`: floor-like orientation plus flat, slab-like appearance cues
  when RGB is present.
- `plane_unknown`: geometry is planar enough for fitting but subtype evidence is
  incomplete, conflicting, or below threshold.

Subtype confidence should stay separate from primitive confidence. A plane can
be geometrically strong while semantically ambiguous.

## Report Contract

V1 should add a new optional report:

```text
plane_detections.json
```

Recommended top-level fields:

- `schema_version`
- `image_path`
- `empty_room_image_path`
- `vggt_depth_path`
- `vggt_points_path` or `background_mesh_path`
- `image_width`
- `image_height`
- `camera`
- `planes`
- `model_info`

Recommended per-plane fields:

- `id`
- `label`
- `primitive_label`
- `plane_subtype`
- `bbox_xyxy`
- `mask_polygon`, `mask_path`, or `mesh_face_indices`
- `center_xyz`
- `normal_xyz`
- `normal_confidence`
- `plane_extent_xyz` or `bbox_3d`
- `inlier_ratio`
- `coverage_ratio`
- `visibility_ratio`
- `residual_mm`
- `support_count`
- `needs_review`
- `source`
- `failure_reason`

`label` may use the subtype or a display-oriented label, but downstream object
placement must depend on geometry fields and quality metadata, not semantic
text alone.

## CLI Surface

The runtime switch should be additive and off by default until the plane detector
is implemented and tested. In the new pipeline, this switch consumes the
empty-room VGGT background artifacts produced by the background lane.

Proposed flags for the future empty-room background geometry command:

- `--detect-planes`
- `--plane-confidence-threshold`
- `--plane-min-area-px`
- `--plane-subtypes wall,floor,ceiling,road,concrete_floor`

Recommended defaults:

- `--detect-planes`: disabled;
- `--plane-confidence-threshold`: `0.55`;
- `--plane-min-area-px`: at least `2%` of image area or a fixed floor such as
  `1024`, whichever is larger;
- `--plane-subtypes`: `wall,floor,ceiling,road,concrete_floor`.

When disabled, object proposal behavior is unchanged. When enabled, the
empty-room background lane should write `background/planes.json` beside the
other background geometry artifacts.

Do not wire this through the archived primitive-proxy public calls; that
execution path is retired.

## Placement Contract

Background VGGT geometry remains the plane authority. Plane subclasses should
map to explicit placement metadata such as `plane_subtype`, `center_xyz`,
`normal_xyz`, support extents, quality fields, and snap eligibility.

The exact metadata location can be refined during implementation, but the rule
is fixed: new plane work must feed object placement and mesh snapping directly,
not the retired primitive-proxy fit report.

If a plane is low confidence:

- mark `needs_review=true`;
- include a specific `failure_reason`;
- optionally exclude it from final placement if its geometry confidence is below
  the configured threshold;
- do not delete or rename existing object detections.

## Quality Policy

Each plane candidate should report:

- `inlier_ratio`;
- `residual_mm`;
- `support_count`;
- `coverage_ratio`;
- `visibility_ratio`;
- `normal_confidence`;
- `needs_review`;
- `failure_reason`.

Suggested failure reasons:

- `insufficient_area`
- `low_inlier_ratio`
- `high_residual`
- `normal_ambiguous`
- `subtype_ambiguous`
- `fragmented_support`
- `occlusion_too_high`
- `depth_missing`

Review policy:

- strong plane geometry plus weak subtype evidence should keep the plane and use
  `plane_subtype=plane_unknown`;
- weak plane geometry should be marked `needs_review` and may be excluded from
  exported placement;
- no plane detector output should break object proposal or object mesh stages.

## Test Plan

Unit tests:

- classify `wall`, `floor`, `ceiling`, `road`, and `plane_unknown` from
  synthetic normals and frame positions;
- validate connected-component area filtering and thin-strip rejection;
- verify quality gates for residual, area, normal confidence, and fallback
  reasons;
- ensure unsupported subtype names are rejected by config parsing.

Integration tests:

- synthetic room scene with known floor, wall, and ceiling planes writes
  `plane_detections.json` with stable fields;
- road-like slab in the lower frame classifies as `road` only when geometry and
  appearance criteria both support it;
- noisy-depth and partial-occlusion cases emit deterministic `needs_review` and
  `failure_reason` values;
- object proposal output keeps `detect/detections.json` unchanged when
  plane detection is disabled.

Acceptance tests:

- enabling plane detection adds `background/planes.json` without changing
  object proposal reports;
- plane candidates expose snap targets for object meshes;
- weak or ambiguous planes are reviewable through `needs_review` and
  `failure_reason`;
- no retired primitive-proxy output is required.

## Implementation Order

1. Add config parsing and report type definitions without changing default
   behavior.
2. Implement point-cloud extraction and candidate plane fitting from depth.
3. Add mask cleanup and connected-component filtering.
4. Add subtype tagging and quality fields.
5. Thread the optional plane report into the empty-room background lane.
6. Add object-snap consumers that read plane geometry directly.
7. Add focused unit tests, then integration tests using synthetic assets.
