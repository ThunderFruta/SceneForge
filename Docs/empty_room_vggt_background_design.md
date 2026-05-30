# Empty Room VGGT Mesh Design

This document defines the narrow V1 background lane for SceneForge: produce a
good empty-room image and an inspectable VGGT-derived room mesh from one original
image plus SAM3/GroundingDINO-SAM3 object masks.

The output of this lane is background geometry only. It does not place objects,
snap meshes to planes, run original-image object VGGT, or compose the final
scene. Those steps belong in separate later docs after the empty-room mesh is
trustworthy.

## Objective

Create a clean empty-room background that preserves the original camera,
perspective, walls, floor, ceiling, lighting, and material style, then run VGGT
on that cleaned image and export a usable background mesh for inspection in
Blender.

OpenAI image editing is used only to fill removed foreground regions with
plausible room surfaces. VGGT is the geometry source for the mesh.

## Non-Goals

This V1 intentionally does not include:

- original-image VGGT object placement;
- object OBB fitting;
- Hunyuan3D or TripoSR mesh alignment;
- support-plane selection;
- object-to-plane snapping;
- final scene composition;
- retired primitive-proxy outputs or fallbacks.

Plane detection can consume this lane later, but the first useful deliverable is
a visually and geometrically inspectable empty-room mesh.

## Pipeline

1. Run object proposals on the original image.

   Use SAM3 or GroundingDINO-SAM3 to write the existing proposal artifacts:
   `detections.json`, `overlay.png`, and per-object masks under the object
   output folder. These masks identify movable foreground content to remove.

2. Build the foreground removal mask.

   Combine selected object masks into one full-frame mask. Expand it enough to
   remove object borders, shadows, contact patches, and small occluders. Exclude
   structural proposals such as walls, floors, ceilings, windows, doors, roads,
   or background surfaces unless a later explicit override says otherwise.

   Proposals with weak masks or `mask_quality=rectangular_fallback` should be
   excluded by default or recorded as review-required in the metadata.

3. Prepare the OpenAI edit input.

   Create a full-frame copy of the original image with the selected foreground
   regions removed. Use transparency when the edit backend supports alpha.
   Otherwise use a neutral or black fill that is also recorded in metadata. Keep
   the original image size and framing exactly.

4. Inpaint the empty room.

   Ask the image edit backend to fill the removed regions as empty room
   surfaces, preserving the original camera, perspective, lighting, walls,
   floor, ceiling, trim, windows, doors, and material style. The prompt should
   explicitly remove movable foreground objects and avoid adding replacement
   furniture, platforms, rugs, props, or new room layout features.

5. Run VGGT on the empty-room image.

   Use `background/empty_room.png` as the VGGT input. Export depth, points,
   camera data, confidence when available, and a sampled mesh. Normalize the
   exported geometry into the SceneForge camera-space contract:

   - `X` points image right;
   - `Y` points away from the source camera along depth;
   - `Z` points image up.

6. Export the empty-room mesh for inspection.

   Write a mesh artifact that can be opened directly in Blender. The mesh should
   preserve the VGGT point-map source and include enough sampled detail to judge
   whether major walls, floor, ceiling, openings, and hidden inpainted surfaces
   are coherent. V1 can use OBJ first; GLB or `.blend` export can be added after
   the mesh contract is stable.

7. Write quality metadata.

   Record enough information to answer whether this is a usable empty-room mesh:
   what was removed, how much was inpainted, which VGGT artifacts were produced,
   whether the output resolution/framing stayed fixed, and what warnings require
   manual review.

## Artifacts

Recommended background artifacts:

- `background/empty_room.png`
- `background/empty_room_openai_input.png`
- `background/empty_room_openai_mask.png`
- `background/empty_room_mask.png`
- `background/empty_room_metadata.json`
- `background/vggt_depth.png`
- `background/vggt_depth.npy`
- `background/vggt_points.npy`
- `background/vggt_points.xyz`
- `background/vggt_camera.json`
- `background/vggt_confidence.png`, when available
- `background/vggt_geometry.json`
- `background/empty_room_mesh.obj`
- optional `background/empty_room_mesh.glb`
- optional `background/mesh_preview.png`

Avoid writing object placement or snapping artifacts from this lane.

## Report Contracts

`background/empty_room_metadata.json` should include:

- source image path;
- source detections path;
- source object mask directory;
- selected removed detection IDs;
- excluded detection IDs and reasons;
- protected structural labels;
- mask quality counts;
- review-required detections;
- mask coverage ratio;
- mask expansion settings;
- OpenAI input image path;
- OpenAI mask image path;
- output image path;
- image edit backend, model, and prompt;
- whether the original image was supplied as reference context;
- fill mode before editing: transparent, neutral, or black;
- resolution/framing preservation status;
- warnings such as possible object reappearance, structural hallucination, or
  excessive masked area.

`background/vggt_geometry.json` should include:

- empty-room image path;
- VGGT backend, model, device, cache, and local-only settings;
- depth, point, camera, confidence, and mesh artifact paths;
- image width and height;
- coordinate contract in SceneForge camera space;
- point count and valid-point ratio;
- sampled mesh vertex/face counts;
- confidence summary when available;
- known transform used for Blender/OBJ export;
- warnings and `needs_review` status.

The mesh report should make it clear whether vertices come from VGGT world
points, camera-space depth, or another explicit source. Hidden changes to axis
orientation are not acceptable; any axis conversion must be recorded.

## Quality Policy

The empty-room mesh is usable when:

- `empty_room.png` has the same resolution and framing as the original image;
- removed objects do not visibly reappear;
- inpainted regions continue the surrounding room surfaces instead of inventing
  unrelated furniture or layout changes;
- VGGT writes depth, points, camera, and mesh artifacts for the empty-room image;
- the mesh opens upright and source-facing in Blender using the documented axis
  conversion;
- large structural surfaces are coherent enough for manual inspection;
- warnings are explicit when image edit quality or VGGT confidence is weak.

The run should mark `needs_review=true` when:

- too much of the image is masked;
- the mask includes likely structural regions;
- rectangular fallback masks are used;
- OpenAI changes the camera framing or room layout;
- removed objects reappear;
- VGGT produces sparse, unstable, or low-confidence geometry;
- the exported mesh has invalid bounds, extreme scale, or wrong orientation.

Weak output should be preserved for inspection with clear metadata. It should
not fall back to retired primitive-proxy execution.

## CLI Shape

The public flow can run as two staged commands:

```bash
python3 run.py generate-empty-room \
  --image Output/Latest/render/image.png \
  --detections Output/Latest/detect/detections.json \
  --objects Output/Latest/objects \
  --output Output/Latest/background

python3 run.py run-vggt \
  --image Output/Latest/background/empty_room.png \
  --output Output/Latest/background \
  --backend vggt \
  --mesh-stem empty_room_mesh \
  --vggt-cache-dir Models/Geometry/VGGT/hf-cache \
  --vggt-local-only
```

The existing `run-vggt` command can be reused for the second stage if it writes
the same VGGT artifacts under `background/`. Use `--mesh-stem empty_room_mesh`
when writing the background-lane mesh artifacts.

Or as one combined command:

```bash
python3 run.py run-empty-room-vggt \
  --image Output/Latest/render/image.png \
  --detections Output/Latest/detect/detections.json \
  --objects Output/Latest/objects \
  --output Output/Latest/background \
  --empty-room-backend openai-image \
  --vggt-repo-dir Models/Geometry/VGGT/repo \
  --vggt-cache-dir Models/Geometry/VGGT/hf-cache \
  --vggt-local-only \
  --mesh-stem empty_room_mesh
```

`construct-empty-room` is an alias for the same combined command.

Optional future empty-room flags:

- `--mask-dilation-px`
- `--mask-feather-px`
- `--include-detection-id`
- `--exclude-detection-id`
- `--allow-rectangular-fallback-masks`
- `--empty-room-backend openai-image`
- `--review-only`

## Testing Plan

Unit tests:

- selected detection masks combine into one deterministic full-frame mask;
- mask expansion is deterministic and bounded;
- protected structural labels are excluded by default;
- rectangular fallback masks are excluded or marked review-required;
- empty-room prompt construction includes framing and no-furniture constraints;
- metadata records mask coverage, fill mode, selected IDs, and warnings.

Integration tests with fakes:

- fake detections plus a fixture image write `empty_room_mask.png`,
  `empty_room_openai_input.png`, `empty_room_metadata.json`, and placeholder
  `empty_room.png`;
- fake VGGT on `empty_room.png` writes deterministic depth, points, camera,
  geometry report, and mesh artifacts under `background/`;
- output image dimensions match the source dimensions;
- weak masks and excessive mask coverage produce `needs_review=true`;
- no object placement, snapping, or primitive-proxy artifacts are written.

Acceptance checks:

- the command sequence produces a background folder that can be inspected without
  running any object-placement stage;
- the empty-room image is faithful enough for VGGT to see the room structure;
- the mesh opens in Blender with the expected orientation and scale envelope;
- reports explain whether the empty-room mesh is ready for later plane detection.

## Open Implementation Notes

- Prioritize artifact clarity over one-command convenience.
- Keep the OpenAI edit stage and VGGT mesh stage independently rerunnable.
- Store the exact prompt and image-edit inputs so bad inpaints can be debugged.
- Prefer deterministic fake backends for tests so the mesh contract can stabilize
  before depending on model output quality.
- Once the empty-room mesh is consistently good, plane detection can consume the
  same `background/` VGGT artifacts in a separate design.
