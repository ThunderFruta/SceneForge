# Hybrid Pipeline Wish List Plan

This plan captures the highest-priority work for a confidence-gated hybrid pipeline:
fuzzy-to-detail reconstruction with a coarse depth scaffold first, then selective
detail only where confidence is high.

## Pivot: Fuzzy to Detail

SceneForge should start with a robust, coarse geometric scaffold that captures
the big room/object shape before adding finer geometry. Fine detail should be
layered onto the scaffold only when the supporting depth, silhouette, and region
evidence is strong enough.

SAM-style segmentation should be used for segment proposals and boundaries, not
as a trusted semantic classifier. SceneForge labels remain an internal geometry
decision layer.

## Goal

Ship a stable structured reconstruction path that:

- always returns usable coarse geometry,
- improves object quality when confidence is high,
- falls back safely when AI output is uncertain.

## Phase 1: Stable Base Geometry (Depth-First)

1. Keep depth-driven structured mesh as the required base output.
2. Improve base mesh boundary quality with small cleanup, horn rejection, and
   large occlusion gap detection/reporting rather than full large-hole filling.
3. Keep deterministic defaults for reproducibility and test coverage.

Exit criteria:

- Base scene exports reliably without AI stages.
- No severe mesh cracks on sample fixtures.

Implemented foundation:

- `Geometry/Cleanup/` is the cleanup subsystem boundary.
- `Geometry/DepthValidity/` separates invalid/no-data depth from valid far-depth surfaces.
- Structured mode can clean small segmentation mask defects, cap small internal mesh holes, reject obvious spike faces, and report large occlusion gaps in `metrics.json`.
- Structured mode defaults to treating exact black depth as invalid while preserving near-black stable surfaces such as dark back walls.
- Giant holes caused by occlusion are explicitly not filled in Phase 1; they become later candidate-completion work.

## Phase 1.5: Visibility and Completion Split

Do not treat every missing surface as a cleanup hole. Split the failure modes:

1. False invalid-depth/background removal:
   - valid far surfaces can be dark in the depth map,
   - exact black/no-data should remain invalid by default,
   - near-black stable surfaces should survive unless threshold mode is requested.
2. True occluded geometry:
   - surfaces hidden behind objects need priors, multi-view input, or candidate completion.
3. Low-confidence object/detail geometry:
   - rough detail should remain gated and inspectable instead of becoming horn artifacts.

Next major milestone after depth validity repair: primitive candidate completion for room layout and object regions.

## Phase 2: Object Candidate Generation

1. Use segmentation masks to define region proposals and boundaries.
2. Generate multiple per-object geometry candidates:
   - primitive fit candidate (sphere, ellipsoid, cylinder, box),
   - optional AI 2D->3D candidate.
3. Normalize candidate scale and alignment into scene coordinates.

Exit criteria:

- At least two candidates can be produced for target object regions.
- Candidate generation failures do not stop full-scene export.

## Phase 3: Confidence-Gated Selection

1. Score each candidate with a weighted objective:
   - silhouette agreement,
   - depth agreement,
   - smoothness/curvature sanity,
   - contact/support consistency with base scene.
2. Choose the highest-score candidate only when confidence exceeds threshold.
3. Fallback policy:
   - low confidence -> keep base depth mesh or primitive candidate.

Exit criteria:

- Low-confidence regions stop producing horn/transparent carve artifacts.
- Selection is deterministic for fixed inputs and config.

## Phase 4: Composition and Cleanup

1. Insert selected object meshes into the base scene.
2. Run cleanup:
   - reject horn/spike artifacts,
   - close only small defects,
   - report large occlusion gaps instead of hallucinating hidden geometry,
   - normals/UV consistency checks.
3. Export `.blend` and optional `.obj` sidecar as usual.

Exit criteria:

- Merged scenes remain Blender-importable.
- Boundary discontinuities are reduced on known sample scenes.

## Phase 5: User Controls and Quality Modes

1. Add `quality` presets (`fast`, `balanced`, `ultra`) that control tile size, overlap, refinement steps, and texture resolution.
2. Add manual object overrides (`--prior sphere`, `--force-opaque`, etc.).
3. Keep all overrides optional; defaults should remain simple.

Exit criteria:

- Users can trade speed vs detail through one high-level control.
- Manual overrides can fix obvious semantic misses quickly.

## Non-Goals (for this phase)

- End-to-end single-model scene reconstruction as the only path.
- Real-time constraints.
- Fully automatic perfect backside geometry from one image.
