from __future__ import annotations

import math


Vec3 = tuple[float, float, float]


def fit_plane(points: list[Vec3]) -> tuple[Vec3, Vec3] | None:
    """
    Least-squares plane fit via PCA.

    Returns (centroid, normal) where the normal is the eigenvector of the
    scatter matrix with the smallest eigenvalue, oriented to face the camera
    (camera is at the origin; the scene occupies negative Y).

    Returns None when fewer than 3 points are supplied.
    """
    if len(points) < 3:
        return None

    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    cz = sum(p[2] for p in points) / n
    centroid: Vec3 = (cx, cy, cz)

    a: list[list[float]] = [[0.0] * 3 for _ in range(3)]
    for p in points:
        d = (p[0] - cx, p[1] - cy, p[2] - cz)
        for i in range(3):
            for j in range(3):
                a[i][j] += d[i] * d[j]

    eigenvalues, eigenvectors = _jacobi3(a)

    min_idx = min(range(3), key=lambda i: eigenvalues[i])
    normal = eigenvectors[min_idx]

    # Orient normal toward the camera (origin).  Camera direction from centroid
    # is -centroid; flip the normal if it points away.
    if _dot(normal, (-cx, -cy, -cz)) < 0:
        normal = (-normal[0], -normal[1], -normal[2])

    return centroid, normal


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _jacobi3(a: list[list[float]]) -> tuple[list[float], list[Vec3]]:
    """
    Jacobi eigenvalue algorithm for a 3×3 symmetric matrix.

    Returns (eigenvalues, eigenvectors) where eigenvectors[i] corresponds to
    eigenvalues[i].  Eigenvectors are column vectors stored as rows of the
    accumulated rotation matrix.
    """
    m = [[a[i][j] for j in range(3)] for i in range(3)]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]

    for _ in range(100):
        max_off = 0.0
        p, q = 0, 1
        for i in range(3):
            for j in range(i + 1, 3):
                if abs(m[i][j]) > max_off:
                    max_off = abs(m[i][j])
                    p, q = i, j

        if max_off < 1e-12:
            break

        diff = m[q][q] - m[p][p]
        theta = math.pi / 4 if abs(diff) < 1e-12 else 0.5 * math.atan2(2.0 * m[p][q], diff)
        c, s = math.cos(theta), math.sin(theta)

        new_m = [[m[i][j] for j in range(3)] for i in range(3)]
        new_m[p][p] = c * c * m[p][p] - 2 * s * c * m[p][q] + s * s * m[q][q]
        new_m[q][q] = s * s * m[p][p] + 2 * s * c * m[p][q] + c * c * m[q][q]
        new_m[p][q] = new_m[q][p] = 0.0
        for r in range(3):
            if r != p and r != q:
                new_m[r][p] = new_m[p][r] = c * m[r][p] - s * m[r][q]
                new_m[r][q] = new_m[q][r] = s * m[r][p] + c * m[r][q]
        m = new_m

        new_v = [[v[i][j] for j in range(3)] for i in range(3)]
        for r in range(3):
            new_v[r][p] = c * v[r][p] - s * v[r][q]
            new_v[r][q] = s * v[r][p] + c * v[r][q]
        v = new_v

    eigenvalues = [m[i][i] for i in range(3)]
    eigenvectors: list[Vec3] = [tuple(v[r][i] for r in range(3)) for i in range(3)]  # type: ignore[misc]
    return eigenvalues, eigenvectors
