# Empty Room VGGT Background Design

This document defines the planned empty-room background lane for the new
SceneForge pipeline.

The goal is to remove foreground objects from the original frame, generate a
clean empty-room image that preserves the original camera and room layout, run
VGGT on that cleaned background to recover structural planes, then use a second
VGGT pass on the original image to place SAM3/Hunyuan objects back onto those
planes.

## Summary

This is a separate background/structure process that works beside the object
lane.

```text
Object lane:
  original image -> SAM3 masks -> object completion -> Hunyuan3D meshes

Background lane:
  original image + SAM3 masks -> empty-room inpaint -> VGGT -> background mesh -> planes

Placement lane:
  original image -> VGGT object geometry -> snap objects to background planes
```

OpenAI image generation is used for texture and hidden-surface completion. It is
not the geometry authority. VGGT is the geometry source for the empty-room
background and for original-image object placement.

## Intended Flow

1. Run SAM3 or GroundingDINO-SAM3 on the original image.

   This writes the existing proposal artifacts, including `detections.json` and
   per-object masks under the object output folder.

2. Build one full-frame foreground removal mask.

   Combine SAM3 masks for movable foreground objects. Protect structural labels
   such as `wall`, `floor`, `ceiling`, `road`, `plane`, `room`, and
   `background` only when label, mask size/location, and geometry cues agree.
   Expand the mask enough to remove object edges, shadows, contact patches, and
   small occluders. Proposals marked `mask_quality=rectangular_fallback` or
   otherwise weak should be excluded by default or force downstream
   `needs_review` metadata.

3. Prepare the OpenAI empty-room input image.

   Create a full-frame copy of the original image where every selected SAM3
   foreground mask is removed before it is sent to OpenAI. The removed regions
   should be transparent for edit APIs that support alpha, or black/neutral for
   APIs that require an opaque image. Keep the original resolution and framing.
   Save this as `background/empty_room_openai_input.png`.

4. Inpaint an empty room from the masked frame.

   Use OpenAI image edit/inpaint mode with `empty_room_openai_input.png`, the
   combined removal mask, and the original image as reference context when the
   backend supports multiple inputs. The prompt must preserve exact camera
   framing, perspective, lighting, walls, floor, ceiling, trim, windows, doors,
   and material style. The model should fill the transparent/black masked
   regions as empty room surfaces and remove furniture, objects, and foreground
   clutter naturally. If the original image is used as reference context, the
   metadata and acceptance checks must record that removed objects did not
   reappear.

5. Run VGGT on the empty-room image.

   This produces the background geometry substrate: depth, points, camera data,
   and/or a background mesh. This is where structural surfaces come from.

6. Extract planes from the empty-room VGGT geometry.

   Detect floor, wall, ceiling, road, concrete floor, or unknown structural
   planes from the background mesh or point cloud. Plane detection should not run
   from raw original-image SAM3 masks.

7. Run VGGT on the original image.

   Use the original object-filled frame to estimate where the SAM3 objects
   actually sit in camera space. For each object mask, sample original VGGT
   points and fit an object OBB, footprint, and likely contact region.

8. Reconstruct object detail meshes.

   Use the existing object completion and Hunyuan3D path to generate object
   meshes from SAM3 crops. Hunyuan3D provides visual/detail geometry, not scene
   placement authority.

9. Snap objects to background planes.

   Align each object OBB/detail mesh to the original-image VGGT placement, then
   snap its support/contact region to the nearest compatible empty-room plane.
   Furniture should prefer floor/support planes. Wall planes should only be used
   for wall-mounted objects when label and geometry support that choice.

10. Compose the final scene.

   Export a background mesh/textured plane scene plus placed object meshes. Keep
   reports that show which object snapped to which plane and why.

## Artifacts

Recommended background artifacts:

- `background/empty_room.png`
- `background/empty_room_openai_input.png`
- `background/empty_room_mask.png`
- `background/empty_room_reference.png`
- `background/empty_room_metadata.json`
- `background/vggt_depth.png` or equivalent depth artifact
- `background/vggt_points.*` or equivalent point artifact
- `background/vggt_camera.json`
- `background/background_mesh.*`
- `background/planes.json`

Recommended original-object VGGT artifacts:

- `objects_vggt/vggt_depth.png` or equivalent depth artifact
- `objects_vggt/vggt_points.*` or equivalent point artifact
- `objects_vggt/vggt_camera.json`
- `objects_vggt/object_geometry.json`

Recommended alignment artifacts:

- `scene_alignment.json`
- final composed `.blend`
- optional alignment/debug overlay showing object masks, support planes, and snap
  directions.

## Report Contracts

`background/empty_room_metadata.json` should include:

- source image path;
- source detections path;
- source object mask directory;
- OpenAI model and prompt;
- whether the original image was supplied as reference context;
- protected labels;
- removed detection IDs;
- mask quality counts and review-required detections;
- mask coverage ratio;
- mask expansion settings;
- OpenAI input image path;
- output image path;
- whether masked regions were transparent, black, or neutral;
- warnings, including possible object reappearance.

`background/planes.json` should include:

- background VGGT source artifacts;
- camera/coordinate contract in SceneForge camera space: X right, Y depth
  away from camera, Z up;
- plane IDs;
- plane subtype such as `floor`, `wall`, `ceiling`, `road`,
  `concrete_floor`, or `plane_unknown`;
- plane center, normal, extents, support area, confidence, and quality fields;
- optional mesh/texture references for the background surface.

`objects_vggt/object_geometry.json` should include:

- detection ID;
- SAM3 mask source;
- point count and valid-point ratio;
- object OBB center, extents, and rotation in SceneForge camera space;
- contact/footprint estimate;
- point-source and mask-quality metadata;
- quality and failure or review reason.

`scene_alignment.json` should include:

- detection ID;
- selected support plane ID;
- pre-snap object transform;
- post-snap object transform;
- snap delta;
- snap confidence;
- empty-room/original-image VGGT camera and scale reconciliation status;
- whether the object needs review;
- reason for failure when no compatible plane is found.

## Placement Rules

- Preserve the original camera/framing in `empty_room_openai_input.png` and
  `empty_room.png`; do not generate a new plausible room with a different layout.
- Send OpenAI the image after selected SAM3 masks have been transparented,
  blacked out, or neutral-filled, not the untouched original as the main target.
- Use empty-room VGGT for background geometry and planes.
- Use original-image VGGT for object placement.
- Normalize empty-room and original-image VGGT artifacts into SceneForge camera
  space before comparing or snapping: X right, Y depth away from camera, Z up.
- Use Hunyuan3D meshes as detail assets aligned to VGGT-derived OBBs.
- Snap object bases to planes only after the original-image VGGT object pose is
  estimated.
- Do not let OpenAI-generated pixels directly decide object depth or placement.
- Flag weak plane fits, weak object VGGT crops, rectangular fallback masks,
  camera/scale disagreement, and large snap deltas as `needs_review`. Do not
  route failed placement into retired primitive-proxy fallbacks.

## CLI Shape

A future public flow could expose this as explicit staged commands or as an
opt-in reconstruction mode.

Suggested staged commands:

```bash
python3 run.py generate-empty-room   --image Output/Latest/render/image.png   --detections Output/Latest/detect/detections.json   --objects Output/Latest/objects   --output Output/Latest/background

python3 run.py reconstruct-background   --image Output/Latest/background/empty_room.png   --geometry-backend vggt   --output Output/Latest/background

python3 run.py fit-object-placements   --image Output/Latest/render/image.png   --detections Output/Latest/detect/detections.json   --objects Output/Latest/objects   --background-planes Output/Latest/background/planes.json   --geometry-backend vggt   --output Output/Latest/alignment
```

A later scene assembly mode can orchestrate the same steps behind explicit
flags, but the first implementation should keep artifacts staged and inspectable.

## Testing Plan

Unit tests:

- combined mask generation from multiple SAM3 polygons;
- protected-label filtering for structural labels;
- deterministic mask dilation/feathering coverage;
- empty-room prompt construction;
- support-plane selection and snap-delta calculation from fake plane/object
  geometry.

Integration tests with fakes:

- fake detections plus fixture image write `empty_room_mask.png`,
  `empty_room_openai_input.png`, `empty_room_metadata.json`, and placeholder
  `empty_room.png`;
- fake background VGGT output writes deterministic `planes.json`;
- fake original-image VGGT output writes deterministic object OBBs in
  `objects_vggt/object_geometry.json`;
- fake Hunyuan meshes align to fake OBBs and record selected support planes in
  `scene_alignment.json`;
- missing compatible support plane keeps the object and marks it `needs_review`;
- rectangular fallback masks propagate review-required placement metadata.

Acceptance checks:

- `empty_room.png` preserves original resolution and framing;
- planes are derived from empty-room VGGT artifacts;
- objects are placed from original-image VGGT artifacts;
- Hunyuan meshes are aligned after object placement, not before;
- reports make every snap decision inspectable;
- camera/scale disagreement between empty-room and original-image VGGT produces
  weak alignment instead of forced snaps.

## Open Implementation Notes

- The first version should prioritize artifact clarity over one-command
  convenience.
- Background texture quality can be improved independently from placement
  quality because OpenAI inpainting and VGGT geometry are separate stages.
- If empty-room VGGT and original-image VGGT disagree strongly on camera scale or
  orientation, the run should mark alignment as weak instead of forcing a snap.
- The active fallback for weak masks, VGGT, or snapping is explicit failure or
  `needs_review` metadata, not retired primitive-proxy execution.
