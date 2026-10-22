"""Build a fake test set of near-duplicate variants from a few seed images."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_seeds(src: Path) -> list[tuple[str, Image.Image]]:
    seeds = []
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            seeds.append((p.stem, Image.open(p).convert("RGB")))
        except Exception as e:
            print(f"skip {p.name}: {e}")
    if not seeds:
        raise SystemExit(f"No images found in {src}")
    return seeds


def random_crop(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    scale = rng.uniform(0.7, 0.95)
    cw, ch = int(w * scale), int(h * scale)
    left = rng.randint(0, max(0, w - cw))
    top = rng.randint(0, max(0, h - ch))
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.Resampling.BILINEAR)


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    out = img.copy()

    if rng.random() < 0.8:
        out = random_crop(out, rng)

    if rng.random() < 0.5:
        out = ImageOps.mirror(out)

    if rng.random() < 0.7:
        angle = rng.uniform(-12, 12)
        out = out.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(128, 128, 128))

    if rng.random() < 0.7:
        out = ImageEnhance.Color(out).enhance(rng.uniform(0.6, 1.4))
        out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.75, 1.25))
        out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.75, 1.3))

    if rng.random() < 0.4:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.5)))

    return out


def main():
    ap = argparse.ArgumentParser(description="Create ~N near-duplicate variants from seed images")
    ap.add_argument("--src", required=True, help="Folder of ~10 seed images")
    ap.add_argument("--out", required=True, help="Output folder for variants")
    ap.add_argument("--n", type=int, default=300, help="Number of variants to create")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    seeds = load_seeds(src)
    rng = random.Random(args.seed)
    print(f"Loaded {len(seeds)} seeds → writing {args.n} variants to {out}")

    rows: list[tuple[str, str]] = []
    for i in range(args.n):
        label, base = seeds[rng.randrange(len(seeds))]
        variant = augment(base, rng)
        name = f"variant_{i:04d}.jpg"
        variant.save(out / name, quality=90)
        rows.append((name, label))

    labels_path = out / "labels.csv"
    with labels_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        w.writerows(rows)

    print(f"Done: {args.n} images + {labels_path.name} in {out}")


if __name__ == "__main__":
    main()
