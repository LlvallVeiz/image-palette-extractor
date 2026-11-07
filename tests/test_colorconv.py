#!/usr/bin/env python3
"""Assert-based tests for the colour-space conversions.

Run:  python3 tests/test_colorconv.py
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from palette.colorconv import lab_to_srgb, srgb_to_lab  # noqa: E402


def lab_of(rgb):
    return srgb_to_lab(np.array([[rgb]], dtype=np.float64))[0, 0]


# Published sRGB/D65 reference values (Lindbloom): red -> (53.24, 80.09, 67.20)
lab = lab_of((255, 0, 0))
assert np.allclose(lab, [53.241, 80.092, 67.203], atol=0.05), lab

# white -> L=100, a=b=0
lab = lab_of((255, 255, 255))
assert np.allclose(lab, [100.0, 0.0, 0.0], atol=0.05), lab

# black -> L=0, a=b=0
lab = lab_of((0, 0, 0))
assert np.allclose(lab, [0.0, 0.0, 0.0], atol=1e-6), lab

# mid grey is neutral: a and b ~ 0, L ~ 53.6 for #808080
lab = lab_of((128, 128, 128))
assert abs(lab[1]) < 0.05 and abs(lab[2]) < 0.05, lab
assert abs(lab[0] - 53.585) < 0.05, lab

# round trip sRGB -> Lab -> sRGB on a spread of colours (incl. the
# sample image's ground-truth colours) is lossless to within rounding
rng = np.random.default_rng(7)
colors = rng.integers(0, 256, size=(256, 3)).astype(np.float64)
colors[:5] = [[62, 124, 177], [107, 127, 142], [76, 159, 112],
              [224, 192, 132], [242, 209, 107]]
back = lab_to_srgb(srgb_to_lab(colors))
err = np.abs(np.rint(back) - colors).max()
assert err <= 1, f"round-trip error {err}"

print("All tests passed.")
