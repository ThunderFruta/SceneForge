# AGENTS.md

Guidance for coding agents working in this repository.

## Project Context

SceneForge is an early-stage computer graphics project for turning 2D images into usable 3D assets and scenes. The current source of truth is `BEFORE_README.md`, which describes the idea, first milestone, and longer-term roadmap.

The repository was reset on 2026-05-24. The old depth-to-mesh prototype was deleted and replaced with a fresh Python CLI prototype for segmenting a 2D image, labeling each detected object with a closest primitive shape, and fitting those detections to rough geometric 3D proxies from synthetic depth.

Also read:

- `project_preferences.md` for naming and project conventions.
- `structure.md` for the intended repository layout.
- `current_changes.md` for recent documentation and project changes.

## Current Goal

Keep early work focused on the current primitive-detection prototype:

1. Load a 2D image.
2. Segment visible objects.
3. Classify each object as a closest primitive approximation: sphere, cylinder, cone, box, plane, or unknown.
4. Write `detections.json`.
5. Write `overlay.png`.
6. For synthetic depth inputs, fit detections to simple geometric 3D primitives and export a Blender scene.

Prefer a visible, practical MVP over perfect reconstruction or large architecture.

## Engineering Preferences

- Read `BEFORE_README.md` before making design or implementation decisions.
- Use PascalCase for directory names and snake_case for file names.
- Keep changes small and easy to replace while the project shape is still forming.
- Prefer simple, inspectable algorithms before adding heavy ML dependencies.
- Use existing, well-supported libraries for image IO, mesh processing, and export when they are clearly useful.
- Do not introduce a framework, package manager, or large dependency stack without a concrete implementation need.
- For large plan-based refactors, spawn subagents when it will improve speed, parallelism, or review coverage.
- Document any new setup or run commands in a README once real code exists.

## Agent Collaboration

- Use subagents whenever a task has independent parallel work that can improve speed or review quality.
- Good subagent lanes include docs/tree verification, test coverage review, visual acceptance checks, isolated subsystem implementation, and risk review.
- Give subagents clear file or responsibility ownership.
- Do not let multiple subagents edit overlapping files unless the ownership split is explicit.
- Keep final integration, conflict resolution, and end-to-end verification in the main agent thread.

## Suggested First Implementation Shape

For the current code, keep the first pass shaped around:

- A small CLI that accepts an image path, local YOLO segmentation weights, a local CLIP model directory, and an output directory.
- Lazy imports for heavy ML dependencies so fake-backend tests run without model files.
- Deterministic fake backends for tests.
- JSON and overlay outputs before any 3D export work.
- Geometric-only 3D primitive fitting for synthetic depth maps, keeping unknown detections as box proxies.

## Testing And Verification

Once code exists again, prioritize tests that confirm:

- Image loading rejects missing and invalid files.
- Primitive labels stay fixed for V1.
- Detection reports serialize deterministically.
- Fake backend pipeline writes `detections.json` and `overlay.png`.
- Primitive fitting writes `primitive_fits.json` and `fitted_scene.blend` from fake detections plus synthetic depth.
- CLI options produce deterministic output for small fixtures.

## Boundaries

Avoid jumping ahead into RigForge, AvatarForge, or ElasticForge until SceneForge has a usable image-to-object-to-geometric-primitive fitting path.
