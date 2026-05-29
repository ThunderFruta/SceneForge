# Plane Detection Design

This document defines the V1 plane-first path for large structural scene
surfaces such as walls, floors, ceilings, roads, concrete slabs, and other
dominant planar regions.

The goal is to add broad surface coverage without changing the current
SceneForge fitting contract. Plane detection should produce fit-ready plane
evidence from aligned depth first. Semantic subclass labels are metadata and
guidance only; primitive fitting still treats these surfaces as
`primitive_label=plane`.

## Problem

Object proposal detectors are a poor primary source for large structural
surfaces. A wall, floor, ceiling, or road can span most of the frame, be
partially occluded by foreground objects, and have weak visual texture. These
surfaces need a geometry-first pass over the full depth frame instead of only
object-level proposal masks.

SceneForge needs this path to:

- recover large support surfaces that object segmentation may miss;
- keep foreground object detection independent from structural surface fitting;
- preserve the existing `detections.json` and `primitive_fits.json` contracts;
- expose quality fields so bad or ambiguous planes are reviewable.

## Scope

V1 detects connected planar regions from aligned depth geometry across the full
frame. RGB may refine subclass labels, but RGB must not be required for geometry
acceptance.

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
- changing SceneForge's camera coordinate contract.

## Coordinate Contract

Plane detection uses the existing SceneForge camera-space contract from
`SceneGeometry/coordinate_contract.py`:

- `X` points image right;
- `Y` points away from the source camera along depth;
- `Z` points image up;
- depth is aligned with RGB and uses the current white-close, black-far depth
  convention;
- units match the existing primitive fitting and Blender export path.

Plane fit reports must remain compatible with the current primitive fitting
pipeline. A floor, wall, road, or ceiling is still a plane primitive. The subtype
is advisory metadata.

## Pipeline

The proposed sequence is:

1. Normalize inputs.

   Build a valid-depth mask from the aligned depth image. Optionally add RGB
   confidence weighting when RGB is available and stable, but do not require RGB
   for candidate extraction.

2. Extract a point cloud.

   Unproject valid pixels through the current `PinholeCamera` logic so candidate
   generation and primitive fitting share the same camera-space frame.

3. Generate plane candidates.

   Use iterative RANSAC or depth patch clustering with normal support. Each
   accepted candidate must record the plane equation, normal, inlier mask,
   residuals, and support count.

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

6. Emit fit-ready plane objects.

   Plane candidates should be written as enriched detection candidates with
   `primitive_label=plane`, optional `plane_subtype`, and quality metadata.

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
- `depth_path`
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
- `mask_polygon` or `mask_path`
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

`primitive_label` must be `plane` for fit-compatible plane entries. `label` may
use the subtype or a display-oriented label, but fitting must not depend on it.

## CLI Surface

The runtime switch should be additive and off by default until the plane detector
is implemented and tested.

Proposed flags for `reconstruct-scene` and `fit-primitives`:

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

When disabled, existing behavior is unchanged: `fit-primitives` consumes plane
entries from `detections.json` if they already exist.

When enabled, `reconstruct-scene` should write `detect/plane_detections.json`
beside `detect/detections.json`. `fit-primitives` should consume the optional
plane report in addition to existing detections.

## Fitting Contract

Primitive fitting remains the geometry authority.

Plane subclasses map to:

```json
{
  "primitive_label": "plane",
  "fit_quality": {
    "plane_subtype": "floor"
  }
}
```

The exact metadata location can be refined during implementation, but the
compatibility rule is fixed: exported primitive geometry and existing consumers
must still see a plane primitive.

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
- no plane detector output should break the legacy plane path.

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
- `reconstruct-scene` keeps `detect/detections.json` unchanged when
  `--detect-planes` is disabled.

Acceptance tests:

- `fitted_scene.blend` generation is stable with plane detection disabled;
- enabling plane detection adds plane candidates without breaking object fits;
- `primitive_fits.json` keeps `primitive_label=plane` while carrying subtype
  metadata;
- existing quality-gate commands still operate on the final fit report.

## Implementation Order

1. Add config parsing and report type definitions without changing default
   behavior.
2. Implement point-cloud extraction and candidate plane fitting from depth.
3. Add mask cleanup and connected-component filtering.
4. Add subtype tagging and quality fields.
5. Thread the optional plane report into `reconstruct-scene`.
6. Let `fit-primitives` consume the optional report and attach subtype metadata
   to plane fits.
7. Add focused unit tests, then integration tests using synthetic assets.
