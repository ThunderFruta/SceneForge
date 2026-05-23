# Segmentation

SceneForge uses segmentation as an optional decision layer before structured
geometry. The goal is to separate scene areas into labels that geometry can
handle differently.

## Modes

- `none`: current depth-only structured reconstruction.
- `mask`: load a user-provided RGB label mask.
- `auto`: deterministic dependency-free fallback. This is not SAM 3.

## Manual Mask Colors

```text
wall     -> red     (255, 0, 0)
floor    -> green   (0, 255, 0)
ceiling  -> blue    (0, 0, 255)
object   -> yellow  (255, 255, 0)
detail   -> cyan    (0, 255, 255)
unknown  -> black   (0, 0, 0)
```

Unknown colors map to `unknown`. Mask dimensions must match the source image
and depth map.

## Geometry Mapping

- `wall`, `floor`, and `ceiling` become fitted plane regions.
- `object`, `detail`, `unknown`, and valid non-plane labels become detail relief regions.
- `--details` still adds the coverage relief fallback behind structured parts.

## SAM 3 Direction

SAM 3 should be added as a provider under `Segmentation/Providers/SAM3/` after
manual masks prove that mask-guided reconstruction improves output. It should
produce the same `SegmentationMask` type as manual and heuristic providers.
