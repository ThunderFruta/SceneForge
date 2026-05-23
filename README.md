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
- `--output Output`
- `--resolution 128`
- `--depth-strength 1.0`
- `--obj`
- `--no-texture`

## Development

Install test dependencies with:

```bash
pip install -e .[dev]
```

Run tests with:

```bash
pytest
```
