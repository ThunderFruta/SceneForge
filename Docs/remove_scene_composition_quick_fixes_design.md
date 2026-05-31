# Remove Scene-Specific Composition Quick Fixes

## Summary

Replace the current chair, table, and room quick fixes with general placement and alignment logic.

The priority is maximum generality, even if the current sample scene temporarily regresses. The target implementation should remove label- and scene-specific behavior and rely on reusable evidence from masks, VGGT points, mesh geometry, fitted planes, projection diagnostics, and support-contact checks.

## Implementation Goal

Remove scene-specific composition hacks and replace them with generic evidence-based placement: object-agnostic yaw search, generic stable support-contact snapping, and plane/camera-informed VGGT room alignment.

## Key Changes

### Remove chair-specific fixes

- Delete seat-group median scaling.
- Delete table-facing chair rotation.
- Delete front-chair quarter-turn correction.
- Delete chair-specific yaw candidate lists.
- Replace these with object-agnostic yaw search scored by projected mask fit, bbox fit, VGGT visible-point fit, support contact, and collision penalties.

### Remove table-specific floor snap fixes

- Delete the `stable_floor` table-only branch.
- Use one generic stable-contact estimator for all floor-supported objects.
- Fall back to raw mesh bottom only when stable contact evidence is unavailable.

### Remove hand-tuned VGGT room sizing

- Stop fitting room size only from object placement bounds and margin constants.
- Use fitted empty-room floor and wall planes plus camera geometry to align room scale, floor height, wall direction, and usable floor footprint.
- Report the room alignment transform and evidence instead of hiding scale/translation constants in composition code.

### Keep general improvements

- Uniform object scaling.
- Aspect-ratio-preserving mesh normalization.
- VGGT textured empty-room background.
- Fitted planes for support reasoning.
- Explicit diagnostics in placement and scene alignment reports.

## Report and API Changes

Add `orientation_search` to placement reports:

- `yaw_candidates`
- `selected_yaw`
- `loss_breakdown`
- `fallback_reason`

Add generic `support_contact` diagnostics:

- `selection_method`: `stable_footprint`, `raw_bottom`, or `unavailable`
- `vertex_ratio`
- `footprint_span`
- `area_ratio`
- `selected_layer`
- `selected_quantile`

Add `background.room_alignment` diagnostics:

- `method`
- `floor_plane_id`
- `wall_plane_ids`
- `usable_floor_bounds`
- `applied_transform_gltf`

Remove or deprecate scene-specific report fields:

- `seat_group_adjustment`
- front-chair correction notes
- label-specific chair yaw debug fields

## Test Plan

- Assert chair and table labels do not trigger special placement code paths.
- Assert all placed objects keep uniform scale.
- Assert stable-contact snapping is shared by all floor-supported objects.
- Assert raw-bottom snapping is only fallback behavior.
- Assert orientation search records deterministic candidates, selected yaw, and loss breakdown.
- Assert `compose-scene` still writes `scene.glb` and `scene_alignment.json`.
- Assert the current sample still composes `6/6`, but do not assert exact chair yaw.

## Assumptions

- Temporary visual regression is acceptable while removing hacks.
- No new heavyweight ML dependency is required.
- Replacement logic should use existing masks, VGGT points, mesh vertices, projection diagnostics, and fitted planes.
- Code cleanup should happen after this design doc is added.
