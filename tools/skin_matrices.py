"""Column-major affine matrices used by the static and animated skin exporters."""
import math


def mat_mul(a, b):
    """Column-major 4x4 product ``a * b``."""
    out = [0.0] * 16
    for column in range(4):
        for row in range(4):
            out[column * 4 + row] = sum(a[k * 4 + row] * b[column * 4 + k] for k in range(4))
    return out


def try_invert_affine(m):
    """Return the affine inverse, or None for a singular or non-finite determinant."""
    a = lambda row, column: m[column * 4 + row]
    det = (a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1))
           - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0))
           + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)))
    # RageSkeleton.cs uses this absolute cutoff, which allows mirrored bind transforms.
    if not math.isfinite(det) or abs(det) < 1e-12:
        return None
    k = 1.0 / det
    cofactors = (
        (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) * k,
        (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) * k,
        (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) * k,
        (a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)) * k,
        (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) * k,
        (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) * k,
        (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)) * k,
        (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) * k,
        (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) * k,
    )
    tx, ty, tz = a(0, 3), a(1, 3), a(2, 3)
    out = [0.0] * 16
    for row in range(3):
        for column in range(3):
            out[column * 4 + row] = cofactors[row * 3 + column]
        out[12 + row] = -(cofactors[row * 3] * tx
                          + cofactors[row * 3 + 1] * ty
                          + cofactors[row * 3 + 2] * tz)
    out[15] = 1.0
    return out
