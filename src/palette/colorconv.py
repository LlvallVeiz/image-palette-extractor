"""Colour-space conversions: sRGB <-> CIE XYZ <-> CIE L*a*b* (D65).

Implements the standard conversion chain used for the ``--space lab``
clustering mode:

    sRGB (gamma-encoded) -> linear RGB -> CIE XYZ (D65) -> CIE L*a*b*

with the sRGB/D65 matrix and the Lab nonlinearity constants as published
by Bruce Lindbloom (see docs/RESEARCH.md).  All functions are vectorised
and accept arrays of shape (..., 3).
"""

import numpy as np

# Linear sRGB -> CIE XYZ (D65, 2° observer), Lindbloom's matrix.
SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
# CIE XYZ -> linear sRGB, the published inverse matrix.
XYZ_TO_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])

#: CIE D65 reference white (Xn, Yn, Zn), 2° observer.
D65_WHITE = np.array([0.95047, 1.0, 1.08883])

_EPS = (6.0 / 29.0) ** 3        # δ³ with δ = 6/29
_KAPPA = (29.0 / 3.0) ** 3      # κ = (29/3)³


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, 12.92 * c,
                    1.055 * c ** (1.0 / 2.4) - 0.055)


def _f(t: np.ndarray) -> np.ndarray:
    """The Lab nonlinearity: cube root above ε, linear segment below."""
    return np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)


def _f_inv(f: np.ndarray) -> np.ndarray:
    return np.where(f ** 3 > _EPS, f ** 3, (116.0 * f - 16.0) / _KAPPA)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 255] to CIE L*a*b* (D65).

    L* lies in [0, 100]; a*/b* are roughly in [-128, 127].
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = _srgb_to_linear(rgb / 255.0)
    xyz = linear @ SRGB_TO_XYZ.T
    f = _f(xyz / D65_WHITE)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE L*a*b* (D65) back to sRGB values in [0, 255].

    Out-of-gamut results are clipped to [0, 255] (a Lab centroid can sit
    slightly outside the sRGB gamut); the return dtype is float, callers
    round as needed.
    """
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = _f_inv(f) * D65_WHITE
    # Y has a special case in the standard: if L > κ·ε then Y = Yn·fy³,
    # which is exactly what _f_inv yields through the fy channel except
    # when fy³ <= ε but L > κε — the two conditions coincide
    # (L > κε  <=>  fy³ > ε), so no special case is needed.
    linear = xyz @ XYZ_TO_SRGB.T
    return np.clip(_linear_to_srgb(linear), 0.0, 1.0) * 255.0
