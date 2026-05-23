# Hybrid Pipeline Wish List Plan

This plan captures the highest-priority work for a confidence-gated hybrid pipeline:
depth-first global reconstruction plus selective object refinement.

## Goal

Ship a stable structured reconstruction path that:

- always returns usable coarse geometry,
- improves object quality when confidence is high,
- falls back safely when AI output is uncertain.

## Phase 1: Stable Base Geometry (Depth-First)

1. Keep depth-driven structured mesh as the required base output.
2. Improve base mesh boundary quality (mask edge smoothing, seam reduction, tiny-hole fill).
3. Keep deterministic defaults for reproducibility and test coverage.

Exit criteria:

- Base scene exports reliably without AI stages.
- No severe mesh cracks on sample fixtures.

## Phase 2: Object Candidate Generation

1. Use segmentation masks to define object regions.
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
   - penetration checks,
   - boundary stitch/weld,
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
