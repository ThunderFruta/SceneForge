# Project Preferences

These preferences guide early SceneForge work. Update this file as the project becomes more concrete.

## Naming

- Use PascalCase for directory names.
- Use snake_case for file names.
- Keep Markdown filenames in snake_case unless a specific established file name is expected by tools.

Examples:

```text
Source/
  mesh_builder.py
  obj_exporter.py

Tests/
  test_mesh_builder.py
```

## Project Shape

- Keep the first implementation small and easy to inspect.
- Prefer practical computer graphics steps over broad AI architecture at the start.
- Add dependencies only when they directly support the image-to-mesh prototype.
- Keep generated assets, fixtures, and samples clearly separated from source code.

## Documentation

- Keep `BEFORE_README.md` as the early roadmap until a full `README.md` exists.
- Update `current_changes.md` when adding meaningful files, features, or conventions.
- Update `structure.md` when the actual repository layout changes.

## Style

- Favor clear names over abbreviated names.
- Prefer deterministic code paths for prototype features so outputs are easier to test.
- Keep comments short and useful.
