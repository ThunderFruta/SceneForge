# Structure

SceneForge has been reset for a fresh implementation direction. This file describes the current prototype layout.

## Current Files

- `BEFORE_README.md`: early project idea, roadmap, and first milestone.
- `README.md`: current project overview and reset status.
- `requirements.txt`: local Python dependencies for image IO, segmentation, CLIP inference, and tests.
- `run.py`: CLI entrypoint.
- `AGENTS.md`: instructions for coding agents working in this repository.
- `Docs/`: design notes and integration contracts for active or near-term pipeline work.
- `Archives/`: local ignored retired-code archive, when present; not part of the active tracked tree.
- `structure.md`: intended repository structure and naming notes.
- `current_changes.md`: short record of recent project changes.
- `project_preferences.md`: project conventions and preferences.

## Current Layout

Use PascalCase for directories and snake_case for files.

```text
SceneForge/
  AGENTS.md
  BEFORE_README.md
  README.md
  requirements.txt
  run.py
  current_changes.md
  project_preferences.md
  structure.md

  Docs/

  Input/
    Image/

  Segmentation/

  ShapeDetection/

  SceneGeometry/

  ObjectCompletion/

  ObjectReconstruction/

  OutputWriter/

  Output/
    Latest/
    Archive/

  Configs/

  Models/
    Completion/
    Mesh/
      Hunyuan3D/
      TripoSR/
    OpenVocabulary/
      GroundingDINO/
      SAM3/

  Tools/
    Dataset/
    Training/

  Tests/
    CLI/
    Input/
    Pipeline/
    ShapeDetection/
```

## Previous Prototype Removed

The previous Python image-to-mesh prototype has been intentionally deleted, including source modules, tests, generated outputs, sample assets, configuration placeholders, and local environment/cache artifacts.

Do not assume any of the old CLI, package metadata, output formats, or module boundaries still exist. Reintroduce them only if they fit the new direction.

## Current Prototype Responsibilities

- `Input/`: load and validate RGB images.
- `Segmentation/`: open-vocabulary proposal adapters for SAM3, GroundingDINO-SAM3, and RAM/GroundingDINO-SAM3. Retired depth-edge, Primitive3D, RGB YOLO, and RGBD YOLO detector paths are not active.
- `ShapeDetection/`: build proposal reports with detector-neutral object fields and unassigned primitive labels by default.
- `SceneGeometry/`: shared coordinate/FOV/depth contracts for current and future VGGT placement reports.
- `ObjectCompletion/`: complete object crops for downstream object mesh reconstruction.
- `ObjectReconstruction/`: run object-level 3D reconstruction stages such as Hunyuan3D and completed-crop TripoSR mesh export.
- `Runtime/`: backend-neutral runtime helpers such as torch device resolution shared by detector and future 3D model paths.
- `OutputWriter/`: write stable JSON reports, annotated overlay images, depth previews, and metric comparison summaries.
- `Docs/`: preserve active design contracts for SAM3 proposals, empty-room VGGT background reconstruction, and plane detection from the empty-room VGGT model.
- `Archives/`: ignored local-only storage for retired code snapshots, if needed.
- `Output/Latest/`: ignored active run output.
- `Output/Archive/`: ignored timestamped run folders only.
- `Tools/Dataset/`: reserved for active dataset utilities.
- `Tools/Training/`: reserved for active training utilities.
- `Tests/`: verify image loading, schemas, open-vocabulary setup/adapters, active object completion/reconstruction helpers, and deterministic CLI behavior without model weights.

## Possible Future Layout

If SceneForge returns to image-to-mesh export work, a practical expanded layout may look like this:

```text
SceneForge/
  AGENTS.md
  BEFORE_README.md
  README.md
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
    Cleanup/
    DepthValidity/
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

  Segmentation/
    Core/
    Providers/
    Integration/

  Configs/
    App/
    Pipeline/
    Mesh/
    Export/
    Segmentation/

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

Treat this as a candidate layout, not a requirement. Keep the next structure smaller if the new idea does not need these boundaries yet.

Avoid building directories for future roadmap items until SceneForge has a useful SAM3-to-empty-room-VGGT-to-object-mesh scene path.
