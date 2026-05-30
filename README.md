# SceneForge

SceneForge is a computer graphics project for turning 2D images into usable 3D assets and scenes.

The repository was reset on 2026-05-24. The previous Python prototype, generated outputs, sample assets, tests, and local environment artifacts were removed so the next implementation could start cleanly.

For the original project idea and first milestone, read `BEFORE_README.md`.

## Current Prototype

The primitive-proxy reconstruction pipeline is retired and no longer part of the active CLI. Useful implementation ideas may exist in a local ignored `Archives/` checkout or in git history, but the active user path is the staged SAM3/object/VGGT direction.

SceneForge is moving to a staged scene pipeline:

```text
Original image
  -> SAM3 / GroundingDINO-SAM3 object proposals
  -> object crop completion
  -> Hunyuan3D or TripoSR object meshes

Original image + SAM3 masks
  -> foreground mask removal
  -> OpenAI empty-room inpaint
  -> VGGT empty-room background OBJ/GLB mesh

Original image + SAM3 masks
  -> VGGT object placement
  -> snap object meshes to empty-room planes
  -> final composed scene
```

Currently active public pieces:

- open-vocabulary proposal setup and readiness checks for GroundingDINO/SAM3;
- `detect-shapes` for proposal-only `detections.json` and `overlay.png`;
- `generate-empty-room` for empty-room mask/input/image artifacts;
- `construct-empty-room` / `run-empty-room-vggt` for OpenAI/fake empty-room generation plus VGGT OBJ/GLB export;
- `fit-empty-room-planes` for XYZ-aligned floor/wall planes from empty-room VGGT points;
- `run-vggt` for empty-room or original-image VGGT depth, points, OBJ, and GLB mesh artifacts;
- `fit-vggt-boxes` for original-image VGGT object placement boxes;
- `complete-objects` for object crop completion;
- `reconstruct-objects` for Hunyuan3D or TripoSR object meshes;
- `compose-scene` for the first combined background plus placed-object GLB scene.

Retired primitive-proxy public calls have been removed from the active CLI.

V1 uses local model paths only. It does not download weights at runtime.

Local model paths in this workspace:

- Open vocabulary: `Models/OpenVocabulary/GroundingDINO/` and `Models/OpenVocabulary/SAM3/`;
- Hunyuan3D: `Models/Mesh/Hunyuan3D/`;
- TripoSR fallback: `Models/Mesh/TripoSR/`;
- retired/debug model families are not active runtime dependencies.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Usage

Prepare the local GroundingDINO/SAM3 layout without downloading weights automatically:

```bash
.venv/bin/python run.py prepare-open-vocab-layout --root Models/OpenVocabulary
```

Audit the local open-vocabulary setup before real inference:

```bash
.venv/bin/python run.py audit-open-vocab-readiness \
  --root Models/OpenVocabulary \
  --backend groundingdino-sam3 \
  --output Output/Latest/open_vocab_readiness.json
```

Run proposal-only object detection when readiness passes:

```bash
.venv/bin/python run.py detect-shapes \
  --backend groundingdino-sam3 \
  --image path/to/image.png \
  --open-vocab-root Models/OpenVocabulary \
  --text-prompt-preset scene-primitives-v1 \
  --output Output/Latest/detect \
  --device auto
```

Complete object crops after detection has written `Output/Latest/objects`:

```bash
.venv/bin/python run.py complete-objects \
  --objects Output/Latest/objects \
  --completion-backend openai-image
```

Run object-level mesh reconstruction:

```bash
.venv/bin/python run.py reconstruct-objects \
  --objects Output/Latest/objects \
  --backend hunyuan3d \
  --model tencent/Hunyuan3D-2.1 \
  --device auto \
  --source completed
```

Run the empty-room background mesh lane after detection:

```bash
.venv/bin/python run.py construct-empty-room \
  --image path/to/image.png \
  --detections Output/Latest/detect/detections.json \
  --objects Output/Latest/objects \
  --output Output/Latest/background \
  --empty-room-backend openai-image \
  --mesh-stem empty_room_mesh \
  --device auto
```

See `Docs/empty_room_vggt_background_design.md`. Later plane detection is tracked separately in `Docs/plane_detection_design.md`.

Compose the current background, VGGT object placements, and object meshes into one GLB:

```bash
.venv/bin/python run.py fit-empty-room-planes \
  --background Output/Latest/background \
  --output Output/Latest/background

.venv/bin/python run.py compose-scene \
  --background Output/Latest/background/empty_room_planes.glb \
  --objects Output/Latest/objects \
  --object-geometry Output/Latest/objects_vggt/object_geometry.json \
  --output Output/Latest/scene \
  --background-fit raw
```


## Development

```bash
.venv/bin/python -m pytest
```

Generated outputs and the local virtual environment are ignored by git. Keep the active run under `Output/Latest/`. Move older runs under timestamped folders in `Output/Archive/`, such as `Output/Archive/20260524_153000/`.

## Archived Training Notes

Primitive-proxy datasets, RGBD YOLO comparison training, enrichment/fusion, and primitive fitting are retired from the active README. Use git history or a local ignored `Archives/` copy for implementation details if a future plan reactivates them.
