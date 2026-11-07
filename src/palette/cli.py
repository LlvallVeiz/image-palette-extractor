"""Command-line interface: ``extract`` and ``sample`` subcommands."""

import argparse
import sys

import numpy as np
from PIL import Image

from .colorconv import lab_to_srgb, srgb_to_lab
from .imaging import draw_swatch, load_pixels, make_sample_image, rgb_to_hex
from .kmeans import kmeans

_INERTIA_UNITS = {
    "rgb": "squared RGB units (0-255 channels)",
    "lab": "squared CIE76 Lab units (ΔE² summed)",
}


def _extract_entries(pixels_rgb: np.ndarray, result, space: str):
    """Turn a clustering result into sorted palette entries."""
    if space == "lab":
        centers_rgb = lab_to_srgb(result.centers)
    else:
        centers_rgb = result.centers
    counts = np.bincount(result.labels, minlength=len(result.centers))
    entries = []
    for center, count in zip(centers_rgb, counts):
        rgb = tuple(int(round(float(c))) for c in center)
        entries.append({
            "rgb": rgb, "hex": rgb_to_hex(rgb),
            "share": float(count) / len(result.labels),
        })
    entries.sort(key=lambda e: -e["share"])
    return entries


def _cmd_extract(args: argparse.Namespace) -> int:
    pixels, info = load_pixels(args.image, max_pixels=args.max_pixels)
    space_data = srgb_to_lab(pixels) if args.space == "lab" else pixels
    result = kmeans(space_data, k=args.k, seed=args.seed,
                    max_iter=args.max_iter, init=args.init)
    entries = _extract_entries(pixels, result, args.space)

    print(f"Image        : {args.image} "
          f"({info['width']}x{info['height']}, "
          f"{info['pixels_used']}/{info['pixels_total']} pixels used)")
    print(f"Clustering   : k={args.k}, space={args.space}, "
          f"init={args.init}, seed={args.seed}")
    print(f"Iterations   : {result.n_iter}")
    print(f"Final inertia: {result.inertia:,.1f} "
          f"({_INERTIA_UNITS[args.space]})")
    print("Palette      :")
    for e in entries:
        r, g, b = e["rgb"]
        print(f"  {e['hex']}  rgb({r:3d}, {g:3d}, {b:3d})  "
              f"{e['share'] * 100:5.1f}%")

    if args.txt:
        with open(args.txt, "w", encoding="utf-8") as fh:
            fh.write(f"# palette for {args.image}: k={args.k}, "
                     f"space={args.space}, init={args.init}, "
                     f"seed={args.seed}, iterations={result.n_iter}, "
                     f"inertia={result.inertia:.1f}\n")
            for e in entries:
                r, g, b = e["rgb"]
                fh.write(f"{e['hex']}  rgb({r}, {g}, {b})  "
                         f"{e['share'] * 100:.1f}%\n")
        print(f"Wrote {args.txt}")
    if args.out:
        draw_swatch(entries, args.out)
        print(f"Wrote {args.out}")
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    img, truth = make_sample_image(noise=args.noise, seed=args.seed)
    Image.fromarray(img).save(args.out)
    print(f"Wrote {args.out} ({img.shape[1]}x{img.shape[0]}), "
          f"noise sigma={args.noise}")
    print("Ground truth (exact region shares before noise):")
    for t in truth:
        r, g, b = t["rgb"]
        print(f"  {t['name']:<6} {t['hex']}  rgb({r:3d}, {g:3d}, {b:3d})"
              f"  {t['share'] * 100:5.1f}%")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the dominant colours of an image with "
                    "from-scratch k-means (RGB or CIE Lab space)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser("extract", help="extract a palette from an image")
    p_ext.add_argument("image", help="input image (any Pillow format)")
    p_ext.add_argument("--k", type=int, default=5,
                       help="number of colours (default 5)")
    p_ext.add_argument("--space", choices=("rgb", "lab"), default="rgb",
                       help="clustering colour space (default rgb)")
    p_ext.add_argument("--init", choices=("k-means++", "random"),
                       default="k-means++",
                       help="centre initialisation (default k-means++)")
    p_ext.add_argument("--seed", type=int, default=0,
                       help="RNG seed (default 0)")
    p_ext.add_argument("--max-iter", type=int, default=100)
    p_ext.add_argument("--max-pixels", type=int, default=200_000,
                       help="sub-sample images larger than this")
    p_ext.add_argument("--out", default="palette.png",
                       help="swatch image to write (default palette.png)")
    p_ext.add_argument("--txt", default="palette.txt",
                       help="text palette to write (default palette.txt)")
    p_ext.set_defaults(func=_cmd_extract)

    p_sam = sub.add_parser("sample",
                           help="generate the bundled synthetic sample "
                                "image and print its ground truth")
    p_sam.add_argument("--out", default="sample.png")
    p_sam.add_argument("--noise", type=float, default=3.0,
                       help="Gaussian noise sigma added to the flat "
                            "regions (default 3.0)")
    p_sam.add_argument("--seed", type=int, default=42)
    p_sam.set_defaults(func=_cmd_sample)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
