# AGENTS.md

Guidance for coding agents working in this repository.

## Project Context

SceneForge is an early-stage computer graphics project for turning 2D images into usable 3D assets and scenes. The current source of truth is `BEFORE_README.md`, which describes the idea, first milestone, and longer-term roadmap.

The repository now has a Python CLI prototype. It can load image/depth inputs, build relief or structured geometry, export `.blend` by default through Blender, optionally keep OBJ sidecars, render `preview.png`, and use optional segmentation masks for structured mode.

Also read:

- `project_preferences.md` for naming and project conventions.
- `structure.md` for the intended repository layout.
- `current_changes.md` for recent documentation and project changes.

## Current Goal

Keep early work focused on the first SceneForge prototype:

1. Load a 2D image.
2. Use a provided or estimated depth map.
3. Convert depth into a mesh.
4. Project the image as a texture.
5. Export a Blender-friendly asset, starting with `.obj`.

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

When code is added, a reasonable first pass is:

- A small CLI that accepts an image path and optional depth image path.
- A mesh builder that samples pixels on a configurable grid.
- A basic smoothing or decimation option.
- `.obj` export with optional texture material support.
- A tiny sample asset or generated test fixture, if licensing is clear.

## Testing And Verification

For early code, prioritize tests that confirm:

- Image and depth dimensions are handled correctly.
- Mesh vertices, faces, UVs, and normals are generated consistently.
- Exported files can be parsed by a standard library or opened by Blender.
- CLI options produce deterministic output for small fixtures.

## Boundaries

Avoid jumping ahead into RigForge, AvatarForge, or ElasticForge implementation until SceneForge has a usable image-to-mesh export path.
