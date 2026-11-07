"""Image loading, the bundled synthetic sample image, and swatch output."""

import numpy as np
from PIL import Image, ImageDraw

# The sample image is a small synthetic "landscape" built from five flat
# regions with known colours, plus a little seeded noise.  Because the
# generator knows the region map exactly, the true dominant colours and
# their shares are known ground truth for checking the extractor.
SAMPLE_REGIONS: list[tuple[str, tuple[int, int, int]]] = [
    ("sky", (62, 124, 177)),      # #3E7CB1
    ("hills", (107, 127, 142)),   # #6B7F8E
    ("grass", (76, 159, 112)),    # #4C9F70
    ("path", (224, 192, 132)),    # #E0C084
    ("sun", (242, 209, 107)),     # #F2D16B
]


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(float(c))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def load_pixels(path: str, max_pixels: int = 200_000):
    """Load an image as an (n, 3) float64 array of RGB pixels.

    Images with more than ``max_pixels`` pixels are deterministically
    sub-sampled (evenly spaced indices) to keep clustering fast; the
    returned info dict reports both counts.
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).reshape(-1, 3)
    total = len(arr)
    if total > max_pixels:
        idx = np.linspace(0, total - 1, max_pixels).astype(int)
        arr = arr[idx]
    info = {"width": img.width, "height": img.height,
            "pixels_total": total, "pixels_used": len(arr)}
    return arr.astype(np.float64), info


def make_sample_image(width: int = 480, height: int = 320,
                      noise: float = 3.0, seed: int = 42):
    """Build the synthetic sample image and its ground truth.

    Returns (image_uint8 (h, w, 3), ground_truth) where ground_truth is
    a list of dicts {name, rgb, hex, share} sorted by share, computed
    from the exact region map (before noise).
    """
    labels = np.zeros((height, width), dtype=int)
    labels[: height // 2, :] = 0                    # sky: top half
    labels[height // 2: int(height * 0.6), :] = 1   # hills band
    labels[int(height * 0.6):, :] = 2               # grass
    # Path: a trapezoid widening towards the bottom of the grass area.
    top = int(height * 0.6)
    for row in range(top, height):
        t = (row - top) / max(1, (height - top - 1))
        half = int(20 + 60 * t)
        labels[row, width // 2 - half: width // 2 + half] = 3
    # Sun: a disc in the sky.
    yy, xx = np.mgrid[0:height, 0:width]
    sun = (xx - int(width * 0.83)) ** 2 + (yy - int(height * 0.19)) ** 2 \
        <= 28 ** 2
    labels[sun] = 4

    img = np.zeros((height, width, 3), dtype=np.float64)
    truth = []
    for idx, (name, rgb) in enumerate(SAMPLE_REGIONS):
        mask = labels == idx
        img[mask] = rgb
        truth.append({
            "name": name, "rgb": rgb, "hex": rgb_to_hex(rgb),
            "share": float(mask.mean()),
        })
    if noise > 0:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0.0, noise, img.shape)
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    truth.sort(key=lambda t: -t["share"])
    return img, truth


def draw_swatch(entries: list[dict], out_path: str,
                block_h: int = 110, label_h: int = 48,
                total_w: int = 880) -> None:
    """Draw a palette swatch strip: one block per colour with its hex
    code and share underneath. ``entries`` are dicts {hex, rgb, share}.
    """
    k = len(entries)
    block_w = total_w // k
    img = Image.new("RGB", (block_w * k, block_h + label_h), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    for i, e in enumerate(entries):
        x0, x1 = i * block_w, (i + 1) * block_w
        draw.rectangle([x0, 0, x1, block_h],
                       fill=tuple(int(round(c)) for c in e["rgb"]))
        draw.text((x0 + 8, block_h + 6), e["hex"], fill=(255, 255, 255))
        draw.text((x0 + 8, block_h + 24), f"{e['share'] * 100:.1f}%",
                  fill=(200, 200, 200))
    img.save(out_path)
