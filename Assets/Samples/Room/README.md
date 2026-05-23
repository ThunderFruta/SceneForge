# Room Sample

Local extracted room model sample. `room.blend` is the edited working sample and has random per-object materials applied for easier visual separation in Blender.

## Files

```text
OBJ/room.obj
OBJ/room.mtl
ThreeDS/room.3ds
C4D/room.c4d
room.blend
room_render.png
room_render_depth.png
```

These files are large third-party model assets and are ignored by Git. Keep this directory as a local sample/input area unless the asset license and storage plan are made explicit.

## Notes

- Treat `room.blend` as the current editable Blender sample.
- `room_render.png` is a 1600x1000 interior-facing rendered 2D preview from `room.blend`.
- `room_render_depth.png` is a grayscale depth-style render generated from the same camera view.
- `Output/room_reconstructed.blend` is the current SceneForge reconstruction generated from `room_render.png` and `room_render_depth.png`.
- `Output/room_structured.blend` is the current structured-mode reconstruction generated from the same image/depth pair.
- The original extracted OBJ, 3DS, and C4D files are kept only as local source assets.
