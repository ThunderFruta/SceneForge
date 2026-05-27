# SceneForge

SceneForge is a long-term computer graphics project for turning 2D images into usable 3D assets and scenes.

This file is a pre-README: a place to pin the idea before the project has real code, architecture, or setup instructions.

## Core Idea

Start with a single 2D image and generate a rough 3D scene or object that can be opened in Blender.

The first version should not try to solve perfect 3D reconstruction. It should focus on a practical, visible MVP:

1. Load an image.
2. Estimate or provide a depth map.
3. Convert the depth map into a mesh.
4. Project the image onto the mesh as a texture.
5. Export the result as `.obj`, `.glb`, or another Blender-friendly format.

## Long-Term Roadmap

```text
SceneForge
  2D image -> textured 3D scene/environment

RigForge
  static 3D mesh -> rigged mesh with joints

AvatarForge v1
  combine SceneForge + RigForge:
  2D character/object image -> rough 3D mesh -> rigged mesh

ElasticForge
  static 3D mesh -> deformable/elastic controls

AvatarForge v2
  combine everything:
  2D image -> 3D mesh -> rigged + elastic/deformable model
```

## First Milestone

Build the smallest useful SceneForge prototype:

- Input: grayscale depth image or normal image.
- Output: simple textured mesh.
- Export: `.blend` file that opens in Blender.
- Controls: mesh resolution, depth strength, smoothing, and texture on/off.

## Why This Project

This project is meant to be deep enough to grow for years, but scoped enough to start now.

It can teach:

- image processing
- depth maps
- mesh generation
- UVs and texture projection
- Blender formats
- 3D math
- later, rigging and deformation

The goal is not to make a perfect AI model immediately. The goal is to build a pipeline that starts simple and becomes more powerful over time.

