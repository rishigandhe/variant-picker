"""Build a short cold-email walkthrough MP4 from a real picker run.

Usage:
  python make_demo_video.py --folder ./test_variants --out ./demo_video/variant_picker_walkthrough.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import picker

W, H = 1280, 720
FPS = 30
BG = (18, 18, 20)
FG = (245, 245, 242)
MUTED = (160, 160, 155)
ACCENT = (230, 70, 60)
OK = (70, 180, 120)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _new() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


def _text(draw: ImageDraw.ImageDraw, xy, text, size=36, fill=FG, bold=False):
    draw.text(xy, text, font=_font(size, bold=bold), fill=fill)


def _wrap(draw, text, font, max_width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def slide_title(title: str, subtitle: str = "") -> Image.Image:
    img = _new()
    d = ImageDraw.Draw(img)
    _text(d, (80, 250), title, size=54, bold=True)
    if subtitle:
        font = _font(28)
        y = 330
        for line in _wrap(d, subtitle, font, W - 160):
            d.text((80, y), line, font=font, fill=MUTED)
            y += 40
    _text(d, (80, H - 80), "variant-picker", size=22, fill=MUTED)
    return img


def slide_problem() -> Image.Image:
    img = _new()
    d = ImageDraw.Draw(img)
    _text(d, (80, 80), "The problem", size=44, bold=True)
    bullets = [
        "AI campaigns generate hundreds of near-duplicate shots.",
        "Reviewers waste time scrolling past the same sneaker 40 times.",
        "You need the K images that are both good and visually different.",
    ]
    y = 180
    for b in bullets:
        font = _font(28)
        for line in _wrap(d, "•  " + b, font, W - 160):
            d.text((80, y), line, font=font, fill=FG)
            y += 40
        y += 24
    return img


def _thumb_grid(paths: list[str], cols: int = 4, cell: int = 200, gap: int = 12) -> Image.Image:
    rows = (len(paths) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell + (cols - 1) * gap, rows * cell + (rows - 1) * gap), BG)
    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        # center on cell
        canvas = Image.new("RGB", (cell, cell), (30, 30, 32))
        ox = (cell - im.width) // 2
        oy = (cell - im.height) // 2
        canvas.paste(im, (ox, oy))
        grid.paste(canvas, (c * (cell + gap), r * (cell + gap)))
    return grid


def slide_gallery(title: str, paths: list[str], caption: str, accent=False) -> Image.Image:
    img = _new()
    d = ImageDraw.Draw(img)
    color = ACCENT if accent else FG
    _text(d, (80, 40), title, size=40, bold=True, fill=color)
    _text(d, (80, 95), caption, size=24, fill=MUTED)
    grid = _thumb_grid(paths[:12], cols=6, cell=170, gap=10)
    # center grid
    gx = (W - grid.width) // 2
    img.paste(grid, (gx, 150))
    return img


def slide_metrics(result: dict) -> Image.Image:
    img = _new()
    d = ImageDraw.Draw(img)
    _text(d, (80, 60), "How it scores vs random", size=42, bold=True)
    _text(d, (80, 120), result["label_source"], size=24, fill=MUTED)

    n = result["clusters_total"]
    rows = [
        (
            "Clusters covered",
            f"{result['clusters_covered']}/{n}",
            f"random {result['clusters_covered_random']:.1f}/{n}",
        ),
        (
            "Distinct labels in picks",
            f"{result['distinct_labels']}",
            f"random {result['distinct_labels_random']:.1f}",
        ),
        (
            "Missed images",
            f"{result['pct_missed']:.1f}%",
            f"random {result['pct_missed_random']:.1f}%",
        ),
    ]
    y = 200
    for name, picked, rand in rows:
        _text(d, (80, y), name, size=26, fill=MUTED)
        _text(d, (80, y + 40), picked, size=48, bold=True, fill=OK)
        _text(d, (420, y + 55), f"vs {rand}", size=28, fill=MUTED)
        y += 130

    _text(
        d,
        (80, H - 90),
        f"Miss threshold = median NN cosine ({result['miss_threshold']:.3f})  ·  embed: {result['embed_model']}",
        size=20,
        fill=MUTED,
    )
    return img


def slide_how() -> Image.Image:
    img = _new()
    d = ImageDraw.Draw(img)
    _text(d, (80, 80), "How it works", size=44, bold=True)
    steps = [
        "1. Embed every image (CLIP or DINOv2)",
        "2. Drop low-quality shots (sharpness ± prompt match)",
        "3. Greedy facility-location picks K diverse, high-quality images",
        "4. Everything else clusters under its nearest pick",
        "5. Labels score the result — they never affect selection",
    ]
    y = 180
    for s in steps:
        _text(d, (80, y), s, size=28)
        y += 70
    return img


def slide_cta() -> Image.Image:
    img = _new()
    d = ImageDraw.Draw(img)
    _text(d, (80, 240), "Send us 300 variants.", size=48, bold=True)
    _text(d, (80, 320), "We'll show you the 12 a reviewer should actually see.", size=30, fill=MUTED)
    _text(d, (80, H - 80), "variant-picker", size=22, fill=MUTED)
    return img


def write_frames(frames: list[tuple[Image.Image, float]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    idx = 0
    for im, secs in frames:
        n = max(1, int(round(secs * FPS)))
        for _ in range(n):
            p = out_dir / f"frame_{idx:05d}.png"
            im.save(p)
            paths.append(p)
            idx += 1
    return paths


def encode_mp4(frame_dir: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        str(frame_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="./test_variants")
    ap.add_argument("--out", default="./demo_video/variant_picker_walkthrough.mp4")
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--label-column", default=None)
    ap.add_argument("--embed-model", default="CLIP ViT-B-32")
    args = ap.parse_args()

    print("Running picker…")
    result = picker.run(
        folder=args.folder,
        K=args.K,
        quality_percentile=10,
        label_column=args.label_column,
        embed_model=args.embed_model,
    )
    paths = result["paths"]
    picked_paths = [paths[i] for i in result["picked"]]
    random_paths = [paths[i] for i in result["random_idxs"]]

    frames: list[tuple[Image.Image, float]] = [
        (slide_title("variant-picker", "Surface the diverse AI product shots a reviewer should actually see."), 3.5),
        (slide_problem(), 4.5),
        (slide_how(), 5.0),
        (
            slide_gallery(
                "Random sample of K",
                random_paths,
                "What a reviewer often gets: near-duplicates, incomplete range.",
            ),
            4.0,
        ),
        (
            slide_gallery(
                "Picked variants",
                picked_paths,
                "Greedy diversity + quality — labels used only for scoring afterward.",
                accent=True,
            ),
            5.0,
        ),
        (slide_metrics(result), 5.5),
        (slide_cta(), 3.5),
    ]

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        print(f"Writing {sum(int(s * FPS) for _, s in frames)} frames…")
        write_frames(frames, td_path)
        out = Path(args.out).resolve()
        print(f"Encoding {out}…")
        encode_mp4(td_path, out)
        print(f"Done: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
