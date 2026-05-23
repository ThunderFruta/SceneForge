# Tree

Updated: 2026-05-23

Empty directory placeholders (`.gitkeep`) are omitted from this display.

```text
SceneForge/
  .gitignore
  AGENTS.md
  BEFORE_README.md
  pyproject.toml
  run.py
  current_changes.md
  project_preferences.md
  structure.md

  Assets/
    Fixtures/
      tiny_depth.pgm
      tiny_rgb.ppm
    Samples/
      Room/
        README.md
      testing_depth.png
      testing_image.png

  Configs/
    App/
    Export/
    Mesh/
    Pipeline/

  Core/
    __init__.py
    Config/
    Types/
      __init__.py
      mesh_data.py
      scene_data.py
    Utils/
      __init__.py
      output_paths.py

  Docs/
    architecture.md
    tree.md

  Export/
    __init__.py
    Blend/
      __init__.py
      blend_exporter.py
    GLB/
      __init__.py
    OBJ/
      __init__.py
      obj_exporter.py

  Geometry/
    __init__.py
    Mesh/
      __init__.py
      grid_mesh_builder.py
      region_relief_builder.py
    Normals/
      __init__.py
      normal_builder.py
    Planes/
      __init__.py
      masked_plane_mesh_builder.py
      plane_fitter.py
      plane_mesh_builder.py
    Projection/
      __init__.py
      camera_projection.py
    Regions/
      __init__.py
      region_analyzer.py
    Solidify/
      __init__.py
      scan_solidifier.py
    UV/
      __init__.py
      uv_projector.py

  Input/
    __init__.py
    Depth/
      __init__.py
      depth_loader.py
    Image/
      __init__.py
      image_loader.py

  Pipeline/
    __init__.py
    ImageToMesh/
      __init__.py
      image_to_mesh_pipeline.py
    StructuredScene/
      __init__.py
      structured_scene_pipeline.py

  Tests/
    __init__.py
    Core/
      test_output_paths.py
    Export/
      test_blend_exporter.py
      test_obj_exporter.py
    Geometry/
      test_camera_projection.py
      test_plane_fitter.py
      test_grid_mesh_builder.py
      test_normal_builder.py
      test_plane_mesh_builder.py
      test_region_analyzer.py
      test_region_relief_builder.py
      test_scan_solidifier.py
      test_uv_projector.py
    Input/
      test_image_and_depth_loaders.py
    Pipeline/
      test_image_to_mesh_pipeline.py
      test_structured_scene_pipeline.py

  Tools/
    Debug/
    Profiling/
    Scripts/
```
