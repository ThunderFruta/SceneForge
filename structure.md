# Structure

SceneForge has a first Python CLI prototype. This file describes the project layout and where new pieces should fit.

## Current Files

- `BEFORE_README.md`: early project idea, roadmap, and first milestone.
- `AGENTS.md`: instructions for coding agents working in this repository.
- `structure.md`: intended repository structure and naming notes.
- `current_changes.md`: short record of recent project changes.
- `project_preferences.md`: project conventions and preferences.

## Intended Layout

Use PascalCase for directories and snake_case for files.

```text
SceneForge/
  AGENTS.md
  BEFORE_README.md
  pyproject.toml
  run.py
  current_changes.md
  project_preferences.md
  structure.md

  Core/
    Config/
    Types/
    Utils/

  Input/
    Image/
    Depth/

  Geometry/
    Mesh/
    UV/
    Normals/
    Planes/
    Projection/
    Regions/
    Solidify/

  Export/
    Blend/
    OBJ/
    GLB/

  Pipeline/
    ImageToMesh/
    StructuredScene/

  Configs/
    App/
    Pipeline/
    Mesh/
    Export/

  Assets/
    Samples/
    Fixtures/

  Tests/
    Core/
    Input/
    Geometry/
    Export/
    Pipeline/

  Tools/
    Debug/
    Scripts/
    Profiling/

  Docs/
    architecture.md
    tree.md
```

This layout follows the HCRBot pattern of capability-focused top-level modules, subsystem-specific configs, mirrored tests, separate tools, and dedicated architecture/tree docs.

## Module Responsibilities

- `Core/`: shared configuration loading, project types, and small utilities.
- `Input/`: image and depth loading. Keep raw input concerns separate from geometry generation.
- `Geometry/`: mesh, UV, normal, plane, projection, region, solidification, smoothing, and geometry-processing logic.
- `Export/`: output format writers. Use Blender `.blend` as the default user-facing output, write `preview.png` beside each blend, and keep `.obj` as an explicit sidecar/export path.
- `Pipeline/`: orchestration code that wires input, geometry, and export modules together. `ImageToMesh/` is relief mode; `StructuredScene/` is fitted-plane mode with optional detail patches.
- `Configs/`: user-editable settings split by subsystem.
- `Assets/`: samples, generated fixtures, and small test assets.
- `Tests/`: tests that mirror the project modules.
- `Tools/`: debug, profiling, and one-off scripts.
- `Docs/`: architecture notes and generated tree snapshots.

## Example First Files

The first implementation lives inside this structure:

```text
SceneForge/
  Input/
    Image/
      image_loader.py
    Depth/
      depth_loader.py
  Geometry/
    Mesh/
      grid_mesh_builder.py
      region_relief_builder.py
    Normals/
      normal_builder.py
    Planes/
      masked_plane_mesh_builder.py
      plane_fitter.py
      plane_mesh_builder.py
    Projection/
      camera_projection.py
    Regions/
      region_analyzer.py
    Solidify/
      scan_solidifier.py
    UV/
      uv_projector.py
  Export/
    Blend/
      blend_exporter.py
    OBJ/
      obj_exporter.py
  Pipeline/
    ImageToMesh/
      image_to_mesh_pipeline.py
    StructuredScene/
      structured_scene_pipeline.py
  Tests/
    Geometry/
      test_grid_mesh_builder.py
    Export/
      test_obj_exporter.py
```

Do not add `Source/`; keep new code in the capability modules above. Adjust the layout only when a new subsystem has a clear boundary.

## Current Code Area

The current useful code focuses on:

- Loading an image and optional depth map.
- Building a simple grid mesh from depth values in relief mode.
- Building large stable depth regions as masked fitted camera-space plane meshes in structured mode.
- Filtering structured plane/detail faces across configurable depth discontinuities.
- Solidifying structured scene parts with conservative boundary side walls for better off-camera inspection.
- Assigning UV coordinates and per-vertex normals.
- Exporting a Blender-friendly `.obj` file.

Avoid building directories for future roadmap items until SceneForge has a working image-to-mesh prototype.
