# SAM 3 Provider Placeholder

SceneForge does not depend on SAM 3 yet. This directory reserves the adapter
boundary for a future provider.

## Intended Role

Use SAM-style output as segment proposals and boundary hints. Do not trust SAM
class names as SceneForge geometry labels without a local confidence/checking
step.

## Intended Contract

Input:

- RGB source image.
- Optional prompt list.

Possible prompts:

- `wall`
- `floor`
- `ceiling`
- `bed`
- `chair`
- `sofa`
- `table`
- `window`
- `lamp`

Output:

- Segment proposals that can be converted into a `SegmentationMask` using
  SceneForge labels:
  - `wall`
  - `floor`
  - `ceiling`
  - `object`
  - `detail`
  - `unknown`

The SAM 3 provider should live behind the same interface as the manual and
heuristic providers. It should not be wired directly into structured geometry.
