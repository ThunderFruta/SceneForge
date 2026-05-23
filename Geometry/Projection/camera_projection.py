from __future__ import annotations


Vec3 = tuple[float, float, float]
UV = tuple[float, float]


def image_uv(u: float, raw_v: float) -> UV:
    return (u, 1.0 - raw_v)


def project_image_depth_to_point(
    u: float,
    raw_v: float,
    depth: float,
    aspect_ratio: float,
    depth_strength: float,
) -> Vec3:
    distance = 1.0 + (1.0 - depth) * depth_strength
    return (
        (u - 0.5) * aspect_ratio * distance,
        distance,
        (0.5 - raw_v) * distance,
    )


def ray_through_image_point(
    u: float,
    raw_v: float,
    aspect_ratio: float,
) -> Vec3:
    return ((u - 0.5) * aspect_ratio, 1.0, 0.5 - raw_v)


def ray_plane_intersect(
    u: float,
    raw_v: float,
    centroid: Vec3,
    normal: Vec3,
    aspect_ratio: float,
) -> Vec3 | None:
    ray = ray_through_image_point(u, raw_v, aspect_ratio)

    denom = ray[0] * normal[0] + ray[1] * normal[1] + ray[2] * normal[2]
    if abs(denom) < 1e-8:
        return None

    t = (centroid[0] * normal[0] + centroid[1] * normal[1] + centroid[2] * normal[2]) / denom
    if t <= 0:
        return None
    return (ray[0] * t, ray[1] * t, ray[2] * t)
