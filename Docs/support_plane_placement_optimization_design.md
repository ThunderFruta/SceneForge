# Support Plane Placement Optimization Design

This document defines the general placement lane for SceneForge after object
proposal, empty-room reconstruction, structural plane fitting, object completion,
and object mesh reconstruction exist.

The goal is to replace fragile label-specific placement rules with an
evidence-based optimizer that places each reconstructed object mesh into the
empty-room scene using support planes, object masks, original-image VGGT points,
camera projection, and deterministic quality reports.

The immediate motivation is the flower/vase/table failure, but the design is
not specific to flowers or vases. It should work for chairs, tables, lamps,
plants, decor, cups, shelves, wall art, and other objects as long as the
pipeline can produce enough evidence to choose a support mode.

## References

- 3D-RE-GEN repository: https://github.com/cgtuebingen/3D-RE-GEN
- 3D-RE-GEN project page: https://3dregen.jdihlmann.com/

3D-RE-GEN's public description is useful as a design reference because it
combines scene/object reconstruction with constrained placement optimization.
The key idea to adapt is not any object-specific semantic grouping. The key idea
is to fit generated assets back into a reconstructed scene through constrained
degrees of freedom and render/geometry losses.

## Objective

Place reconstructed object meshes into an empty-room scene so they:

- sit on the correct support surface;
- match the original image mask and detection box under camera projection;
- agree with original-image VGGT object points when those points are reliable;
- avoid floating, sinking, penetrating background geometry, or duplicating the
  same physical object;
- produce structured diagnostics that make bad placements easy to inspect.

V1 should be practical, deterministic, and debuggable. It does not need perfect
physical simulation, differentiable rendering, or learned placement prediction.

## Non-Goals

This design intentionally does not include:

- retraining SAM, VGGT, Hunyuan3D, TripoSR, or any detector;
- hard-coded object pair rules such as `vase + flower = composite`;
- automatic semantic scene understanding beyond small support heuristics;
- physically simulated stacking, balance, or articulated pose;
- multi-view reconstruction;
- replacing the raw detector outputs;
- making every bad generated object mesh correct.

The placement optimizer can only fit the mesh it receives. If Hunyuan3D creates
a vase with a floor slab, a bouquet with a table attached, or a chair missing
legs, placement diagnostics should expose the problem instead of hiding it.

## Current Problem

The current composition path has useful pieces but no strong placement contract.
It uses object VGGT boxes, object labels, rough tabletop snapping, and a discrete
projection search. This is enough for simple examples but breaks when:

- the visible object mask includes only part of the physical object;
- the generated mesh includes more or less than the detection mask;
- object VGGT points attach to the wrong visible part;
- a tabletop object uses flower-head depth instead of vase-bottom contact;
- the optimizer improves one bbox edge while making the support contact worse;
- duplicate or overlapping objects are composed independently.

The fix should be a pipeline design: represent evidence, choose a support mode,
optimize with constrained degrees of freedom, and write quality reports.

## Current Code Audit

The repository already has several partial pieces of this design, but they are
not connected through explicit placement contracts yet.

### Existing Empty-Room Plane Code

`SceneGeometry/Planes/empty_room.py` implements `fit_empty_room_planes`.

Current behavior:

- reads `background/vggt_points.npy` and `background/empty_room.png`;
- samples broad image regions for floor, back wall, and right wall evidence;
- exports XYZ-regularized structural planes to `plane_detections.json`;
- exports `empty_room_planes.glb`;
- records plane vertices, normals, subtype labels, fit residuals, and mesh
  stats.

Important gap:

- it only creates large structural planes: `floor`, `back_wall`, and
  `right_wall`;
- it does not detect tabletop, counter, shelf, or support-object planes;
- it does not expose finite support footprints for object placement decisions;
- composer does not consume `plane_detections.json`.

This is good enough for floor/wall staging. It is not enough for general
tabletop placement without a support-object or tabletop-plane stage.

### Existing Original-Image Object Geometry Code

`SceneGeometry/VGGT/regions.py` implements `fit_vggt_boxes`.

Current behavior:

- reads raw detections and per-object masks;
- samples original-image VGGT points inside each object mask;
- writes per-object point diagnostics under `objects_vggt/regions`;
- fits AABB/OBB records into `object_geometry.json`;
- records bbox, mask path/source, point count, valid-point ratio, coverage, and
  review status.

Important gap:

- it writes object boxes, not placement targets;
- it does not choose support mode;
- it does not compare object points to candidate support planes;
- it does not write a compact `object_fit_targets.json`;
- it writes `.xyz`/`.obj` point diagnostics, but not a direct per-object `.npy`
  target point artifact for fast fitting.

This is the right source for object evidence. It should feed a new
`object_fit_targets.json` stage rather than being consumed directly by
`compose-scene`.

### Existing Composer Code

`SceneComposition/composer.py` implements `compose_scene`.

Current behavior:

- reads `object_geometry.json` directly;
- loads per-object reconstructed meshes from `objects/`;
- creates a procedural room-corner background or loads an existing background
  GLB;
- derives floor support from background bounds;
- derives tabletop support from table object mesh top bounds plus 2D bbox
  overlap;
- scales/spreads/orients chairs with label heuristics;
- snaps supported objects onto GLB `Y` support height;
- runs a discrete projection optimizer over horizontal translation, yaw, and
  uniform scale;
- rejects bad projection candidates by vertical bbox edge error;
- records `support_targets`, `support_degrees_of_freedom`,
  `render_to_input_optimization`, `projection_quality`,
  `suppressed_objects`, and `object_overlap_warnings`.

Important gap:

- support targets are internal composer dictionaries, not reusable artifacts;
- support is height-only in GLB coordinates, not a full finite plane;
- no 5-DoF fallback model exists;
- no `--placements` input exists;
- no explicit `object_supports.json`, `object_fit_targets.json`, or
  `object_placements.json` exists;
- no silhouette loss exists;
- no VGGT point-to-mesh loss exists;
- no support-footprint-inside-plane test exists;
- tabletop support relies on object labels plus 2D table overlap, not
  structural plane evidence;
- current 4-DoF search assumes horizontal GLB support planes only.

This means the composer has the seed of Phase 3, but it is doing Phase 1,
Phase 2, Phase 3, and final scene export in one place. The next refactor should
move support selection and fit-target construction out of composer before adding
more optimizer complexity.

## High-Level Pipeline

The placement lane starts after these artifacts exist:

1. Raw object proposals:
   - `detections.json`
   - `overlay.png`
   - per-object masks and crops

2. Empty-room background:
   - `background/empty_room.png`
   - `background/vggt_points.npy`
   - `background/empty_room_mesh.glb`

3. Empty-room structural planes:
   - `background/plane_detections.json`
   - `background/empty_room_planes.glb`

4. Original-image object geometry:
   - `objects_vggt/object_geometry.json`
   - per-object visible VGGT point samples

5. Reconstructed object meshes:
   - `objects/<id>_<label>/hunyuan3d_textured.glb`
   - or `objects/<id>_<label>/triposr_mesh.obj`

The new lane writes:

- `placement/object_supports.json`
- `placement/object_fit_targets.json`
- `placement/object_placements.json`
- `placement/placement_quality.json`
- `scene/scene.glb`
- `scene/scene_alignment.json`
- `scene/input_vs_projection_overlay.png`
- optional debug renders and per-object fit previews

## Design Principle

Use raw proposals as evidence, not authority.

Raw detections should stay unchanged. They are auditable observations. Downstream
placement can decide that an object is unsupported, duplicated, merged upstream,
or not composable, but it should not mutate `detections.json`.

The placement lane owns composition readiness. It should produce its own records
that say:

- which detection produced this placement;
- which mesh was used;
- which support mode was chosen;
- which losses were optimized;
- whether the result is accepted;
- which object needs manual review.

## Coordinate Contract

SceneForge camera space remains:

- `X`: image right;
- `Y`: depth away from source camera;
- `Z`: image up.

GLB export uses the existing conversion:

- GLB `X` maps to SceneForge `X`;
- GLB `Y` maps to SceneForge `Z`;
- GLB `Z` maps to negative SceneForge `Y`.

Placement reports must explicitly state which space each field uses.

Recommended convention:

- fields ending in `_scene` use SceneForge camera space;
- fields ending in `_gltf` use GLB coordinates;
- fields ending in `_px` use source image pixels.

## Evidence Model

Each object placement should be built from an evidence record, not directly from
one bbox.

### Object Evidence

`placement/object_fit_targets.json` should include one record per raw object:

```json
{
  "schema_version": 1,
  "objects": [
    {
      "detection_id": 4,
      "detector_label": "flower",
      "bbox_xyxy_px": [698.0, 49.0, 849.0, 272.0],
      "mask_path": "Output/Latest/objects/04_flower/full_mask.png",
      "mesh_path": "Output/Latest/objects/04_flower/hunyuan3d_textured.glb",
      "visible_points_scene_path": "Output/Latest/objects_vggt/04_flower_points.npy",
      "visible_point_count": 12007,
      "visible_point_valid_ratio": 1.0,
      "mesh_bounds_gltf": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
      "mesh_quality": {
        "has_texture": true,
        "has_large_support_sheet": false,
        "bounds_degenerate": false
      }
    }
  ]
}
```

### Background Evidence

Structural plane records should provide:

- plane ID;
- plane label: `floor`, `tabletop`, `countertop`, `shelf`, `wall`, `ceiling`,
  or `unknown`;
- plane equation in SceneForge space;
- finite polygon or rectangle bounds;
- confidence;
- source pixels or VGGT point indices;
- visual debug mesh path.

Plane support should prefer finite plane extents over infinite plane equations.
An object can sit on a plane only if its projected support footprint overlaps
the finite support area or is close enough to be reviewable.

## Support Modes

Every object gets one support mode before optimization.

### Floor Support

Use for objects that contact the floor:

- chairs;
- tables;
- couches;
- cabinets;
- large plants;
- standing lamps.

Transform model: 4-DoF planar.

Free variables:

- plane-local `u`;
- plane-local `v`;
- yaw around plane normal;
- uniform scale.

Locked variable:

- distance along plane normal is fixed by mesh bottom contact.

### Tabletop Support

Use for small movable objects resting on a table, desk, counter, or shelf:

- cups;
- books;
- bowls;
- vases;
- plants;
- small lamps;
- decor.

Transform model: 4-DoF planar on the selected support object or structural
tabletop plane.

The contact anchor should be the mesh support bottom, not the visible highest
mask points. This matters for bouquets, lampshades, monitors, and tall objects
whose visible points are mostly above the contact surface.

### Wall Support

Use for objects attached to a wall:

- pictures;
- mirrors;
- shelves;
- wall-mounted lights.

Transform model: wall-plane constrained.

Free variables:

- wall-local `u`;
- wall-local `v`;
- in-plane rotation;
- optional uniform scale.

Locked variable:

- distance along wall normal.

### Ceiling Support

Use for hanging lights and ceiling fans when ceiling evidence exists.

V1 can mark this as `needs_review` instead of optimizing it. It is listed here
so the support model does not assume every object belongs to floor/table/wall.

### Floating or Unknown Support

Use 5-DoF fallback only when no reliable support plane exists.

Free variables:

- `tx`;
- `ty`;
- `tz`;
- yaw;
- uniform scale.

The 5-DoF mode is powerful but dangerous. It can make objects float to satisfy
2D projection. It should carry a higher review burden and stricter loss checks.

## 4-DoF Versus 5-DoF

The 4-DoF model is the default for physical objects with a support surface.

In plane-local coordinates:

- `u` and `v` slide the object across the support plane;
- yaw rotates around the plane normal;
- scale changes object size uniformly;
- support height is locked by the plane and mesh contact point.

The 5-DoF model is a fallback:

- `tx`, `ty`, and `tz` translate in scene space;
- yaw rotates around scene up;
- scale changes size uniformly.

5-DoF should not be used just because it gives a lower 2D bbox loss. If 4-DoF
fails, the report should say whether the issue is bad support selection, bad
mesh bounds, wrong object mask, or unreliable geometry.

## Support Selection

Support selection should combine 2D and 3D evidence.

For each object and candidate plane:

1. Compute 2D contact likelihood.
   - object bbox bottom near support bbox top for tabletop;
   - object mask bottom overlaps table/counter/floor projection;
   - object center projects inside finite plane region.

2. Compute 3D contact likelihood.
   - visible object points are above the plane;
   - lower object points are close to the plane;
   - support plane is in front of or near the object's projected region;
   - object depth is compatible with plane depth at that pixel.

3. Compute semantic compatibility.
   - label says tabletop object, floor object, or wall object;
   - support label is compatible;
   - unknown labels do not block geometric evidence.

4. Compute support confidence.
   - plane confidence;
   - object point confidence;
   - mask quality;
   - detection confidence.

The selected support should be the highest scoring compatible plane. If no score
passes threshold, use `unknown_5dof` and mark `needs_review=true`.

## Object Ownership

This design intentionally avoids label-pair composites in the placement stage.

Object ownership should be solved by a separate ownership stage when needed.
That stage can decide whether one raw detection represents:

- one physical object;
- part of another object;
- a duplicate;
- a generated mesh that includes multiple physical objects.

The placement optimizer should consume ownership decisions if they exist:

- `suppressed_by_composite`;
- `source_detection_ids`;
- `relation_role`;
- `composite_id`;
- `placement_group_id`.

But V1 placement should not invent hard-coded composites. It should only avoid
double-composing records that are already suppressed.

## Transform Initialization

Each optimization needs a stable initial transform.

Recommended initialization order:

1. Use object VGGT visible points to estimate center and visible extent.
2. Use support plane to place mesh contact bottom on the plane.
3. Use source mask bbox to initialize image-space scale.
4. Use object label orientation only for broad defaults, such as chairs facing
   the nearest table.
5. If object VGGT points are unreliable, initialize from bbox ray intersection
   with the selected support plane.

For tabletop objects, support contact should use mesh bottom after mesh cleanup.
Do not use bouquet/head/top visible points as the contact authority.

## Loss Functions

The optimizer should report separate losses and a combined score. The combined
score is useful for ranking candidates. The separate losses are necessary for
debugging.

### Silhouette Loss

Render the transformed mesh from the source camera and compare it with the
object mask.

Metrics:

- mask IoU;
- false positive area ratio;
- false negative area ratio;
- contour distance, optional.

This is the most direct 2D placement signal.

V1 can approximate rendering with projected mesh bounds if full silhouette
rendering is too expensive, but the report should label that as a proxy.

### Bbox Projection Loss

Project transformed mesh corners or rendered silhouette bounds into the source
image and compare to the detection bbox.

Metrics:

- bbox IoU;
- center distance normalized by target diagonal;
- area ratio loss;
- top edge error;
- bottom edge error.

The existing projection optimizer already has part of this. It should continue
to reject candidates whose top or bottom edge error is above threshold.

### VGGT Point Loss

Compare transformed mesh surface to original-image VGGT points for the object
mask.

Possible V1 metric:

- sample mesh vertices or surface points;
- compute nearest-neighbor distance to visible object VGGT points;
- trim outliers;
- report median and p90 distances.

This loss should be confidence-weighted. VGGT can be wrong around transparent,
thin, reflective, or textureless objects.

### Support Contact Loss

Measure whether the object actually contacts its selected support plane.

Metrics:

- bottom-contact distance to support plane;
- percentage of bottom footprint over finite support polygon;
- support penetration depth;
- unsupported footprint ratio.

This replaces useless penalties that do not actually measure contact.

### Background Collision Loss

Detect mesh penetration into structural planes or background geometry.

Metrics:

- floor penetration;
- wall penetration;
- table penetration;
- overlap volume with support object or background mesh.

V1 can use AABB/OBB overlap warnings. Later versions can sample signed distance
or plane-side violations.

### Scale Prior Loss

Prevent extreme scale changes that satisfy projection but produce implausible
objects.

Metrics:

- scale ratio from initial estimate;
- label-specific soft range when known;
- mesh bounds sanity.

This should be a soft prior, not a hard semantic dependency.

## Optimization Strategy

V1 should use deterministic discrete search plus local refinement.

For each object:

1. Select support mode.
2. Build initial transform.
3. Search over small candidate grid:
   - plane `u`, plane `v`;
   - yaw;
   - uniform scale.
4. Snap every candidate to the support plane.
5. Score each candidate with visible losses.
6. Reject candidate if hard constraints fail.
7. Keep previous transform if optimization improves score but violates quality.
8. Write full candidate summary.

Discrete search is slower than a closed-form fit but much easier to debug. It
also avoids adding PyTorch/differentiable-renderer dependencies before the rest
of the geometry contract is stable.

Later versions can add:

- differentiable silhouette optimization;
- CMA-ES or scipy local optimization;
- collision-aware refinement;
- multi-object joint optimization.

## Acceptance Policy

An optimized transform is accepted only if all hard checks pass:

- support contact error below threshold;
- top/bottom projection edge error below threshold;
- mesh does not significantly penetrate selected support;
- projected silhouette or bbox remains inside reasonable image bounds;
- scale remains within configured limits;
- loss improves or remains comparable to initial transform.

If a candidate fails hard checks:

- keep the previous support-snapped transform;
- mark `needs_review=true`;
- write `projection_quality.status = rejected` or the relevant rejection reason;
- include the rejected candidate's projected bbox and loss for debugging.

## Duplicate And Overlap Safeguards

The composer should not blindly add every mesh.

V1 safeguards:

- skip records with `suppressed_by_composite`;
- warn on object-object AABB overlap;
- warn on tabletop object overlap;
- warn when two objects have high 2D mask/bbox overlap and compatible labels;
- warn when one generated mesh's projected silhouette covers another object's
  target mask.

Do not automatically merge objects in V1. Suppression and merging should be
owned by an explicit object ownership stage.

## Public CLI Shape

Recommended staged commands:

```bash
python3 run.py choose-object-supports \
  --object-geometry Output/Latest/objects_vggt/object_geometry.json \
  --planes Output/Latest/background/plane_detections.json \
  --detections Output/Latest/detections.json \
  --objects Output/Latest/objects \
  --output Output/Latest/placement
```

```bash
python3 run.py fit-object-placements \
  --supports Output/Latest/placement/object_supports.json \
  --fit-targets Output/Latest/placement/object_fit_targets.json \
  --objects Output/Latest/objects \
  --output Output/Latest/placement
```

```bash
python3 run.py compose-scene \
  --background Output/Latest/background/empty_room_planes.glb \
  --objects Output/Latest/objects \
  --object-geometry Output/Latest/objects_vggt/object_geometry.json \
  --placements Output/Latest/placement/object_placements.json \
  --output Output/Latest/scene
```

`compose-scene` can keep its current fallback behavior when `--placements` is
absent, but the long-term public path should prefer explicit placement records.

## Placement Report Contract

`placement/object_placements.json` should contain:

```json
{
  "schema_version": 1,
  "coordinate_contract": {
    "scene_space": "SceneForge camera space: X right, Y depth away, Z up",
    "gltf_space": "GLB X right, Y up, Z toward camera negative depth"
  },
  "objects": [
    {
      "detection_id": 4,
      "detector_label": "flower",
      "mesh_path": "Output/Latest/objects/04_flower/hunyuan3d_textured.glb",
      "status": "accepted",
      "needs_review": false,
      "support": {
        "mode": "tabletop_4dof",
        "support_plane_id": "plane_table_03",
        "support_detection_id": 3,
        "support_label": "round table",
        "support_confidence": 0.82
      },
      "degrees_of_freedom": {
        "model": "support_plane_4dof",
        "free_parameters": ["plane_u", "plane_v", "yaw_normal", "uniform_scale"],
        "locked_parameters": ["plane_normal_distance"]
      },
      "transform_gltf": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "losses": {
        "total": 0.42,
        "bbox_projection": 0.18,
        "silhouette": null,
        "vggt_points": 0.11,
        "support_contact": 0.0,
        "background_collision": 0.0,
        "scale_prior": 0.03
      },
      "quality": {
        "projection_status": "accepted",
        "support_status": "accepted",
        "collision_status": "accepted",
        "warnings": []
      }
    }
  ]
}
```

## Scene Alignment Contract

`scene/scene_alignment.json` should include placement fields copied from
`object_placements.json`, plus composition-specific outputs:

- `suppressed_objects`;
- `object_overlap_warnings`;
- `projection_quality`;
- `support_targets`;
- `placement_source`;
- `transform_gltf`;
- `source_bounds`;
- `transformed_bounds`;
- `input_vs_projection_overlay`.

The report should answer:

- why this object was placed here;
- what plane supports it;
- which transform model was used;
- whether optimization was accepted;
- what failed when it was rejected.

## Debug Visuals

Placement needs visual QA, but the visuals should be generated from structured
records.

Recommended outputs:

- input image with target masks, target bboxes, and projected fitted mesh bboxes;
- per-object projection preview;
- support plane overlay on empty-room image;
- top-down scene view with object footprints and plane bounds;
- side view showing support contact;
- object overlap debug view.

These should be optional but easy to produce from the placement reports.

## MVP Implementation Plan

### Phase 1: Explicit Support Records

Add `choose-object-supports`:

- read object geometry, detections, object masks, and plane detections;
- classify each object as floor/tabletop/wall/unknown;
- write `object_supports.json`;
- no mesh transform changes yet.

Acceptance:

- chairs choose floor;
- table chooses floor;
- tabletop items choose table/counter when bbox/mask evidence overlaps;
- unknown support objects are marked review.

### Phase 2: Placement Targets

Add `build-object-fit-targets` or fold into `fit-object-placements`:

- collect target bbox;
- collect mask path;
- collect mesh path and mesh bounds;
- collect object VGGT point stats;
- validate missing artifacts deterministically.

Acceptance:

- every accepted object has target bbox, mesh path, support mode, and quality
  warnings if evidence is incomplete.

### Phase 3: 4-DoF Plane Optimizer

Implement support-plane candidate transforms:

- plane-local translation;
- yaw around plane normal;
- uniform scale;
- snap mesh bottom to plane;
- bbox projection loss;
- support contact loss;
- projection rejection.

Acceptance:

- no supported object can float vertically to improve bbox loss;
- bad projection candidates are rejected and reported;
- current composer can consume the placement output.

### Phase 4: 5-DoF Fallback

Add fallback only for unsupported objects:

- scene translation;
- yaw;
- uniform scale;
- stricter review thresholds.

Acceptance:

- 5-DoF records are distinguishable in reports;
- they do not silently replace 4-DoF for floor/tabletop objects.

### Phase 5: Collision And Overlap Metrics

Extend warnings:

- object-object AABB overlap;
- tabletop overlap;
- background plane penetration;
- support footprint outside finite support polygon.

Acceptance:

- `scene_alignment.json` lists deterministic warnings;
- warnings do not block export unless hard-fail flags are added.

### Phase 6: Silhouette Rendering

Replace bbox-only projection with rendered silhouette when practical:

- CPU software render or simple trimesh projection first;
- differentiable rendering later only if needed.

Acceptance:

- silhouette IoU appears in reports;
- bbox projection remains available as a fallback.

## Test Plan

Add small deterministic fixtures:

- floor object with valid support plane;
- tabletop object with valid table plane;
- wall object with valid wall plane;
- object with no support plane falls back to 5-DoF and review;
- optimizer rejects a transform that improves center but ruins top/bottom bbox
  error;
- optimizer rejects support penetration;
- suppressed object is not composed;
- overlap warnings are stable;
- missing mesh creates a failed placement record, not a crash;
- bad object VGGT points reduce confidence but do not erase raw evidence.

CLI tests should verify:

- JSON contracts are stable with sorted keys;
- output paths are deterministic;
- help text does not expose retired primitive fitting;
- commands work on tiny fixtures without heavy model imports.

## Quality Thresholds

Initial thresholds should be conservative and easy to tune:

- projection vertical edge error ratio: `0.35`;
- support contact distance: relative to object height, default `0.03`;
- maximum accepted scale change: `0.5x` to `2.0x`;
- background penetration: default hard reject above small epsilon for planes;
- support footprint outside plane: warning above `0.25`, reject above `0.60`;
- 5-DoF fallback: always `needs_review=true` in V1.

Thresholds should be written into the report so a run is auditable.

## Why This Generalizes

This design is not built around object names. Labels help choose priors, but
geometry controls placement:

- support planes constrain physical contact;
- object masks define image evidence;
- VGGT points define observed 3D evidence;
- generated mesh bounds define contact geometry;
- projection losses verify camera consistency;
- collision losses catch impossible composition.

That means the same system can handle a cup on a table, a chair on the floor, a
picture on a wall, or a lamp on a desk without inventing a new special case for
each object pair.

## Open Questions

- Should support selection use object mask bottom pixels, bbox bottom, or both?
- How much should VGGT point evidence matter for transparent or thin objects?
- Should the object ownership stage run before or after placement targets?
- Should 5-DoF fallback be enabled by default, or only behind a flag?
- What is the cheapest reliable silhouette renderer for local tests?
- Should object meshes be cleaned for support sheets before placement or during
  reconstruction only?

## Recommended Next Step

Implement Phase 1 and Phase 2 first.

Do not start with a complex optimizer. The first useful deliverable is a report
that says, for every object, which support plane it should use, why that plane
was chosen, which evidence was missing, and whether the object is eligible for
4-DoF placement.

Once support records are trustworthy, the 4-DoF optimizer becomes much simpler
and easier to debug.
