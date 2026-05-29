# Structure

SceneForge has been reset for a fresh implementation direction. This file describes the current prototype layout.

## Current Files

- `BEFORE_README.md`: early project idea, roadmap, and first milestone.
- `README.md`: current project overview and reset status.
- `requirements.txt`: local Python dependencies for image IO, segmentation, CLIP inference, and tests.
- `run.py`: CLI entrypoint.
- `AGENTS.md`: instructions for coding agents working in this repository.
- `Docs/`: design notes and integration contracts for active or near-term pipeline work.
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

  PrimitiveFitting/

  OutputWriter/

  Output/
    Latest/
    Archive/

  Configs/
    YOLO/

  Models/
    Edges/
      DexiNed/
    Mesh/
      TripoSR/
    Wireframe/
      HAWP/
    Depth/
      DepthAnythingV3/
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
- `Segmentation/`: detector backend boundary and runtime factory. The active fallback scaffold is depth+edge instance proposal, with `learned_depth_edge_segmenter.py` and `primitive_3d.py` holding the Primitive3D class-agnostic point-cloud instance-mask model seam. Open-vocabulary adapters for GroundingDINO and SAM3 are lazy-loaded proposal backends. YOLO modules remain lazy-loaded legacy/debug/training comparison code.
- `ShapeDetection/`: build detection reports and legacy/fallback primitive label helpers.
- `SceneGeometry/`: shared coordinate/FOV/depth contracts so source renders, detections, enrichment crops, fitting, exports, and metric views use the same frame.
- `ObjectEnrichment/`: build per-object mask/depth/edge/mesh evidence packs; geometry scoring/fusion is the primitive-label authority and detector labels are weak or absent proposal evidence.
- `EdgeDetection/`: provide no-op, simple classical, and local-model dense edge providers.
- `WireframeDetection/`: provide no-op and local HAWP wireframe providers for per-object line/junction evidence.
- `MeshReconstruction/`: provide no-op and local-model advisory mesh candidate providers.
- `ObjectReconstruction/`: run object-level 3D reconstruction stages such as completed-crop TripoSR mesh export.
- `PrimitiveFitting/`: load synthetic depth/enrichment, unproject masked pixels, fit simple geometric 3D primitive proxies, and export Blender scenes.
- `Runtime/`: backend-neutral runtime helpers such as torch device resolution shared by detector, enrichment, and future 3D model paths.
- `OutputWriter/`: write stable JSON reports, annotated overlay images, depth previews, and metric comparison summaries.
- `Docs/`: preserve design contracts for replaceable proposal providers and near-term geometry features such as plane detection.
- `Models/InstanceDetector/`: local learned Primitive3D instance-mask detector checkpoints.
- `Output/Latest/`: ignored active run output.
- `Output/Archive/`: ignored timestamped run folders only.
- `Tools/Dataset/`: generate local synthetic primitive-shape datasets with Blender, write detector-neutral `instance_dataset_manifest.json` files for Primitive3D instance models, and render labeled dataset previews. YOLO label conversion remains for legacy detector experiments.
- `Configs/InstanceDetector/`: detector-neutral configs for the Primitive3D instance-mask model.
- `Tools/Training/`: detector-neutral Primitive3D train/eval from `instance_dataset_manifest.json` plus `Configs/InstanceDetector/`; RGBD YOLO training remains legacy comparison tooling.
- `Tests/`: verify image loading, schemas, test-double pipeline behavior, overlays, and CLI errors without model weights.

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

Avoid building directories for future roadmap items until SceneForge has a useful image-to-object-to-primitive detection prototype.
