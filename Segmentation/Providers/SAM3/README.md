# SAM 3 Provider Placeholder

SceneForge does not depend on SAM 3 yet. This directory reserves the adapter
boundary for a future provider.

## Intended Contract

Input:

- RGB source image.
- Optional prompt list.

Default prompts:

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

- A `SegmentationMask` using SceneForge labels:
  - `wall`
  - `floor`
  - `ceiling`
  - `object`
  - `detail`
  - `unknown`

The SAM 3 provider should live behind the same interface as the manual and
heuristic providers. It should not be wired directly into structured geometry.
