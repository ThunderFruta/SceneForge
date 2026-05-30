# 3D-RE-GEN Model Fitting Reference

This document summarizes how 3D-RE-GEN fits generated object meshes back into a
single-image reconstructed scene, based on a local inspection of:

- `https://github.com/cgtuebingen/3D-RE-GEN`
- `https://3dregen.jdihlmann.com/`
- local inspected commit: `9a6275f3af97eae484c34d0ff5d9cc8177adaa17`

The goal is to capture the useful fitting strategy for SceneForge without
copying 3D-RE-GEN's repo structure or heavy runtime requirements directly.

## Short Version

3D-RE-GEN does not place generated Hunyuan3D meshes by 2D detection boxes alone.
It builds a target object point cloud from the original-image VGGT scene points,
initializes each generated mesh against that object point cloud, then refines the
mesh pose with differentiable rendering and 3D point losses.

For objects that should rest on the floor, it switches to a constrained planar
model: the mesh can slide on the fitted floor plane, yaw around the plane normal,
and scale uniformly, but it cannot freely move off the plane. This is the part
SceneForge should adapt first for support-plane placement.

## Public Pipeline Claim

The project page describes the fitting stage as a differentiable renderer that
optimizes object positions with a 4-DoF ground alignment constraint. In that
constraint, ground-based objects are moved in a local ground-plane frame using:

- 2D translation along the plane;
- yaw rotation around the plane normal;
- uniform scale;
- locked height/contact on the ground plane.

The repo implementation matches that high-level claim with a `PlanarModel` for
floor-supported objects and a less constrained `RegularModel` fallback for
objects not classified as floor-supported.

## Artifacts Feeding The Fitter

The relevant pipeline stages are:

1. `src/segmentation/segmentation.py`

   Runs Grounding DINO plus SAM, filters duplicate detections, writes full-size
   masked object images and cropped object images. It also includes `floor` in
   the default labels, which matters because the later planar fitter looks for a
   floor mask.

2. `src/segmentation/inpaint_nanoBanana.py`

   Uses image editing to create per-object completed/amodal inputs and an
   empty-room image. Its Application-Querying path creates a UI-like image with
   scene context plus an object panel, so the object completion model sees both
   the target and the original scene context.

3. `src/2d_to_3d_models/run.py`

   Runs Hunyuan3D on prepared object images. It removes backgrounds, generates a
   mesh, optionally remeshes/simplifies, applies mesh cleanup, textures the mesh,
   and writes one GLB per object under the configured output folder.

4. `src/camera_and_pointcloud/minimal_demo_vggt.py`

   Runs VGGT on the original image and, when present, the empty-room image. It
   exports camera data, a main scene point cloud, and a separate empty-room point
   cloud. It also applies coordinate conversions and a global `vggt_scene_scale`
   before downstream fitting.

5. `src/scene_reconstruction/source/extract_pc_object.py`

   Recreates each object mask from the full-size masked image, erodes mask edges
   to avoid VGGT boundary noise, samples VGGT scene points inside the mask, filters
   the point cloud, estimates normals, and writes a per-object `.ply` target point
   cloud.

6. `src/scene_reconstruction/source/pose_matching_planar.py`

   Loads the generated object GLB and the matching object target point cloud,
   initializes the mesh pose, chooses floor-constrained or regular fitting, runs
   Adam optimization, and saves the fitted GLB.

7. `src/scene_optimization/scene_optim.py`

   Combines fitted object GLBs into a scene, optionally meshes the empty-room
   background point cloud, creates evaluation point clouds, and can apply
   post-hoc ICP against ground truth when available.

## Initial Mesh Alignment

Before differentiable optimization, 3D-RE-GEN builds a rough pose estimate from
VGGT object points:

1. Load the Hunyuan3D object GLB.
2. Clean the mesh.
3. Load the per-object VGGT point cloud produced from the object mask.
4. Compute an oriented bounding box on the horizontal plane for the target point
   cloud.
5. Compute a matching oriented bounding box for sampled mesh points.
6. Uniformly scale the mesh by comparing target and mesh box volumes.
7. Translate the mesh to the VGGT point-cloud centroid.
8. Optionally run a yaw grid search over candidate rotations and choose the
   lowest Chamfer or point-mesh loss.

This gives the optimizer a reasonable starting pose instead of asking the
renderer to solve position, scale, and orientation from scratch.

## Floor Support Classification

3D-RE-GEN decides whether an object should use the planar floor model with a
simple combination of segmentation and labels:

- it checks whether a file with `floor` in the name exists in the mask folder;
- it computes 2D bbox IoU between the object mask bbox and the floor mask bbox;
- it treats known object-name substrings such as `chair`, `sofa`, `table`,
  `couch`, `bed`, `cabinet`, `desk`, `dresser`, and `plant` as floor objects;
- floor objects use `PlanarModel`;
- other objects use `RegularModel`.

This is useful as a prototype, but SceneForge should not copy the label-only
policy directly. SceneForge's support-plane lane should choose among finite
floor, tabletop, shelf, wall, ceiling, and unknown supports using reportable 2D
and 3D evidence.

## Floor Plane Extraction

For floor-supported objects, `pose_matching_planar.py` extracts a floor point
cloud from VGGT using the floor SAM mask, then compares several plane fits:

- PCA/SVD normal from the floor point cloud;
- RANSAC plus SVD refinement for outlier-heavy floor evidence;
- axis-aligned fallback based on the axis with the lowest variance.

The selected plane writes debugging point clouds:

- original floor points;
- floor residuals colored by distance to the plane;
- sampled points on the ideal plane.

The fitter then builds world-to-plane and plane-to-world transforms where the
plane-local `Y` axis is the plane normal and plane-local `X` and `Z` are tangent
directions.

## Plane-Snapped Initialization

Before optimization, the floor path explicitly puts the object bottom on the
fitted plane:

1. Take the mesh's current world-space bounds.
2. Build the four bottom bbox corners.
3. Project those corners onto the fitted plane.
4. Align the object's up direction to the plane normal with a tilt-only rotation.
5. Translate toward the VGGT object point-cloud centroid.
6. Move the lowest mesh point along the plane normal until it lies on the plane.

This is separate from the optimizer. The optimizer starts with a mesh already in
contact with the support plane.

## Planar 4-DoF Optimizer

`src/scene_reconstruction/source/diff_model_planar.py` defines the floor-aligned
model.

Its trainable parameters are:

- `translation_uv`: two coordinates along the plane;
- `rotation_yaw`: one yaw value around the plane normal;
- `scale`: one uniform scale value.

Its locked parameter is:

- plane-local height/contact. The model constructs final plane-local positions as
  `(u, 0, v)`, forcing the object to remain on the support plane.

Each forward pass:

1. Transforms mesh vertices into plane space.
2. Pivots around the bottom-center of the mesh.
3. Applies uniform scale.
4. Applies yaw around the plane-local normal axis.
5. Applies tangent-plane translation with plane-local height fixed at zero.
6. Transforms vertices back into world space.
7. Renders a silhouette through PyTorch3D.
8. Computes losses and returns the updated mesh.

The core lesson for SceneForge is that support contact is enforced structurally
by the parameterization, not only by a loss term.

## Regular Fallback Optimizer

`src/scene_reconstruction/source/diff_model.py` defines the fallback model for
objects not treated as floor-supported.

Its trainable parameters are:

- 3D translation;
- yaw-only rotation when `use_5DOF` is true;
- full axis-angle rotation when `use_5DOF` is false;
- uniform scale.

Despite the config name, the default `use_5DOF: true` path is essentially:

- 3 translation values;
- 1 yaw value;
- 1 scale value.

This is useful for objects without known support, but it can satisfy 2D losses
by floating or sinking objects. SceneForge should treat this as a review-heavy
fallback, not the default for physical placement.

## Losses

Both models optimize a weighted combination of:

- silhouette loss from a PyTorch3D render against the object mask;
- 3D point-to-mesh distance against the VGGT object point cloud;
- optional background bounding-box penalty.

The planar model uses a Dice plus focal-style silhouette loss and
`point_mesh_face_distance`. The regular model uses Dice plus BCE silhouette loss
and `point_mesh_face_distance`.

The config exposes weights:

- `silhoutte_loss`
- `loss_3d`
- `loss_bbox`

The optimizer is Adam with configurable learning rate, iteration count, gradient
clipping, and early stopping by gradient norm.

## Background Constraint

When `points_emptyRoom.ply` exists, the fitter loads the empty-room VGGT point
cloud, converts it into the fitting coordinate frame, computes its bounds, and
passes a background bbox into the model. The bbox penalty discourages object
vertices from moving outside compatible background extents.

This is a coarse constraint. It is not a real collision model, but it is a useful
early guardrail.

## Final Scene Assembly

After each object is fitted and exported as a GLB, `scene_optim.py` uses
`create_glb_scene` to combine object GLBs into one scene. It can also sample the
combined scene to a point cloud for evaluation and mesh the empty-room point
cloud into a background GLB.

The object fitting stage is therefore per-object and mostly independent. Global
scene assembly happens later and does not appear to jointly optimize all object
placements.

## What SceneForge Should Adapt

The useful parts for SceneForge are:

- build an explicit per-object target point-cloud artifact from original-image
  VGGT points and the object mask;
- initialize generated meshes with target-point OBB/extent/centroid evidence
  before any render optimization;
- make support-constrained transforms structural, especially `u`, `v`, yaw, and
  uniform scale on a finite support plane;
- keep a regular 5-DoF-style fallback only for unsupported or ambiguous objects;
- report losses separately instead of hiding them in one score;
- write debug point clouds, support residuals, projection overlays, and final
  fit reports.

## What SceneForge Should Not Copy Directly

SceneForge should avoid these direct copies:

- one global `floor` mask as the only support source;
- label substring rules as the main support classifier;
- floor-only support modeling;
- hidden intermediate state inside one large pose-matching script;
- heavy PyTorch3D differentiable fitting before report contracts are stable;
- unstructured output folders that make per-stage verification hard.

SceneForge's next placement docs should keep the 3D-RE-GEN idea but generalize it
to `object_supports.json`, `object_fit_targets.json`, `object_placements.json`,
and placement quality reports.

## SceneForge Mapping

The closest SceneForge equivalents are:

- 3D-RE-GEN `fullSize` masks: SceneForge per-object `full_mask.png`.
- 3D-RE-GEN Hunyuan GLBs: SceneForge `objects/<id>_<label>/hunyuan3d_textured.glb`.
- 3D-RE-GEN object point clouds: future SceneForge
  `placement/object_fit_targets.json` plus per-object visible point arrays.
- 3D-RE-GEN floor plane: SceneForge `background/plane_detections.json`, expanded
  from floor/walls to finite support planes.
- 3D-RE-GEN `PlanarModel`: future SceneForge support-plane optimizer.
- 3D-RE-GEN scene GLB combination: SceneForge `compose-scene`.

The most important implementation contract is this:

```text
support choice -> constrained transform model -> projected/rendered fit losses -> explicit acceptance report
```

That contract is more valuable to SceneForge than the specific model stack.
