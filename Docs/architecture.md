# Architecture

SceneForge has a Python CLI prototype for converting an image and optional depth map into a textured Blender file.

```text
Input
  -> Optional segmentation
  -> Geometry relief or structured scene analysis
  -> Export
```

## Module Flow

- `Input/Image/` loads source images as RGB.
- `Input/Depth/` loads grayscale depth maps or derives fallback depth from image luminance.
- `Geometry/Cleanup/` owns deterministic cleanup for structured scans: mask hole/island cleanup, obvious spike rejection, small mesh-hole patching, and large occlusion gap diagnostics.
- `Geometry/DepthValidity/` separates exact invalid/no-data depth from valid near-black far-depth surfaces.
- `Geometry/Mesh/` converts normalized depth values into grid mesh vertices and triangle faces.
- `Geometry/Normals/` computes deterministic per-vertex normals from mesh faces.
- `Geometry/Regions/` finds connected low-variance depth regions for structured mode.
- `Geometry/Planes/` fits large stable regions to camera-space planes and converts their actual masks into textured mesh parts, rejecting faces across configurable depth jumps.
- `Geometry/Projection/` owns the canonical image/depth to 3D coordinate mapping.
- `Geometry/Solidify/` adds conservative boundary side walls to structured scene parts so visible single-view scans have thickness when orbiting off the source camera.
- `Geometry/UV/` generates normalized UV coordinates matching the mesh grid.
- `Segmentation/` owns optional manual-mask and heuristic providers, shared labels, and conversion from labeled masks into structured regions.
- `Export/OBJ/` writes `.obj`, optional `.mtl`, optional texture image files, UVs, and per-vertex normals.
- `Export/Blend/` runs Blender in background mode to import the generated OBJ, save a `.blend`, and render a `preview.png`.
- `Pipeline/ImageToMesh/` wires loading, mesh generation, UV projection, and export together.
- `Pipeline/StructuredScene/` builds fitted planes from image/depth input, with optional detail relief patches.
- `Core/Types/` holds shared data structures such as `MeshData`, `SceneMeshPart`, and `StructuredSceneData`.

## CLI

Run the prototype with:

```bash
python3 run.py \
  --image Assets/Fixtures/tiny_rgb.ppm \
  --depth Assets/Fixtures/tiny_depth.pgm \
  --output Output \
  --mode relief \
  --resolution 2 \
  --depth-strength 1.0 \
  --texture
```

Structured mode:

```bash
python3 run.py \
  --image Assets/Samples/Room/room_render.png \
  --depth Assets/Samples/Room/room_render_depth.png \
  --output Output \
  --mode structured \
  --segmentation mask \
  --mask Assets/Samples/Room/room_render_mask.png \
  --resolution 128 \
  --depth-strength 0.8 \
  --texture
```

Add `--details` in structured mode to include leftover uncertain regions as relief patches plus a valid-depth coverage surface behind the fitted planes. The default structured output is plane-only so visual debugging starts from the stable room surfaces instead of shredded detail fragments.

Use `--segmentation none|mask|auto` to choose the structured-mode segmentation source. `none` keeps depth-only behavior, `mask` loads an RGB label mask, and `auto` uses the dependency-free heuristic provider. `--mask PATH` is only valid with `--segmentation mask`. See `Docs/segmentation.md` for the mask color legend.

Structured mode solidifies by default. Use `--no-solidify` to export front surfaces only, `--solidify` to force side walls, `--solidify-thickness` to tune side-wall depth, and `--depth-edge-threshold` to remove faces across depth discontinuities before solidification.

Structured cleanup is enabled by default. Use `--no-cleanup` to inspect raw structured geometry, `--cleanup` to force cleanup, `--hole-fill-size` to control conservative small-hole filling, and `--spike-threshold conservative|balanced|permissive` to tune horn/spike rejection. Cleanup intentionally does not invent large occluded surfaces; those are reported in `metrics.json` as occlusion gap diagnostics.

Structured depth validity defaults to `--depth-invalid-mode black`, where exact black depth is invalid but near-black depth is preserved as far-surface data. Use `--depth-invalid-mode threshold` with `--min-valid-depth` to restore the old threshold behavior, or `--depth-invalid-mode none` to keep every depth value.

Outputs are written into timestamped run folders under `Output/`, such as `Output/20260523_140506_structured_room_render/room_render.blend`. Each run folder also includes `preview.png` rendered from the source-facing camera for quick resemblance checks. Blender imports are scaled up for easier inspection while preserving SceneForge's canonical coordinates: X right, Y depth away from the camera, and Z up. By default, temporary OBJ files are removed after Blender saves the `.blend`. Add `--obj` to keep a sidecar `.obj`, `.mtl`, and texture image next to the `.blend`.

## Current Boundary

Relief mode intentionally uses a simple grid mesh and nearest-neighbor depth sampling. Structured mode is heuristic: it treats exact black depth as invalid by default, preserves near-black far surfaces, detects large stable depth regions, unprojects those cells into a simple camera-space point cloud, fits a best plane, creates masked textured meshes from the region cells, skips high-depth-jump faces, cleans small mask/mesh defects, rejects obvious spike faces, and adds thin side walls along open boundaries by default. Segmentation can guide structured mode, but the current providers are manual masks and deterministic heuristics, not trained AI. Detail patches and a behind-plane coverage surface are opt-in with `--details`. It does not reconstruct hidden backs, unseen room surfaces, or giant occluded holes from one view. Creating `.blend` output requires Blender on PATH, or a custom path passed with `--blender`.

Keep export formats, depth estimation, and mesh generation separate. Each should remain replaceable without rewriting the full pipeline.
