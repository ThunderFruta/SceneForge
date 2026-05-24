# SceneForge

SceneForge is an early-stage computer graphics project for turning 2D images into simple 3D assets and scenes.

## Current Focus

The first prototype focuses on a practical image-to-mesh export path:

1. Load a 2D image.
2. Use a provided or estimated depth map.
3. Convert depth into a mesh.
4. Project the image as a texture.
5. Export a Blender-friendly asset, starting with `.obj` and `.blend`.

## Usage

The current entrypoint is `run.py`.

Example:

```bash
python run.py --image path/to/image.png --depth path/to/depth.png --mode relief
```

Key options:

- `--mode relief|structured`
- `--segmentation none|mask|auto`
- `--mask path/to/mask.png`
- `--output Output`
- `--resolution 128`
- `--depth-strength 1.0`
- `--depth-invalid-mode black|threshold|none`
- `--min-valid-depth 0.04`
- `--obj`
- `--no-texture`
- `--cleanup` / `--no-cleanup`
- `--hole-fill-size 12`
- `--spike-threshold conservative|balanced|permissive`

Structured segmentation example:

```bash
python run.py \
  --mode structured \
  --segmentation mask \
  --mask Assets/Samples/Room/room_render_mask.png \
  --image Assets/Samples/Room/room_render.png \
  --depth Assets/Samples/Room/room_render_depth.png \
  --cleanup \
  --details
```

Structured cleanup is deterministic and conservative. It fills small mask/mesh defects, removes tiny non-border mask islands, rejects obvious horn/spike faces, and records large occlusion gaps in `metrics.json` instead of inventing hidden room geometry from one view.

Structured depth validity defaults to exact-black invalid handling. That keeps true no-data depth out of the mesh while preserving near-black far surfaces such as dark back walls. Use `--depth-invalid-mode threshold` to discard values below `--min-valid-depth`.

## Viewing .blend outputs

SceneForge does not require opening Blender UI to review output. Use:

```bash
python Tools/Scripts/view_blend.py --blend Output/20260523_.../room.blend
```

The helper now provides:

- Multi-view PNGs in a known prefix: `<blend stem>_view_<view>.png`
- Optional `.glb` with `--gltf` for web/agent viewers
- Optional machine-readable report: `<blend stem>_view_report.json`

Use `--no-gltf` to skip glTF export.

Use view presets to rotate/inspect from different directions:

```bash
python Tools/Scripts/view_blend.py   --blend Output/20260523_.../room.blend   --views front,left,right,top,iso
```

Generate an orbit sweep for camera roll-around inspection:

```bash
python Tools/Scripts/view_blend.py   --blend Output/20260523_.../room.blend   --views orbit   --orbit-steps 12
```

Report options:

- `--report /path/to/report.json` custom output
- `--no-report` disables JSON generation

The JSON report includes mesh stats (`mesh_count`, `total_vertices`, `total_edges`, `total_faces`), bounds center/radius, and per-view camera transforms for easier automated checks.


## Development

Install test dependencies with:

```bash
pip install -e .[dev]
```

Run tests with:

```bash
pytest
```
