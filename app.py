"""Gradio UI for variant-picker — upload a folder of images and Run."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import picker

_STATE: dict = {}


def _scatter_plot(
    embeddings,
    picked,
    random_idxs,
    outlier_idxs=None,
    use_umap: bool = False,
):
    if use_umap:
        try:
            import umap

            xy = umap.UMAP(
                n_components=2, random_state=None, n_neighbors=15, min_dist=0.1, n_jobs=-1
            ).fit_transform(embeddings)
            method = "UMAP"
        except Exception:
            xy = PCA(n_components=2, random_state=42).fit_transform(embeddings)
            method = "PCA (UMAP failed)"
    else:
        xy = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        method = "PCA"

    picked_set = set(picked)
    outlier_set = set(outlier_idxs or [])
    other = [i for i in range(len(xy)) if i not in picked_set and i not in outlier_set]

    fig, ax = plt.subplots(figsize=(7, 5))
    if other:
        ax.scatter(xy[other, 0], xy[other, 1], c="#bbbbbb", s=18, alpha=0.6, label="other")
    if outlier_set:
        oi = list(outlier_set)
        ax.scatter(
            xy[oi, 0],
            xy[oi, 1],
            c="#ff8c00",
            s=55,
            edgecolors="#cc7000",
            linewidths=0.6,
            label="outlier",
            zorder=4,
        )
    if random_idxs:
        ax.scatter(
            xy[random_idxs, 0],
            xy[random_idxs, 1],
            c="#666666",
            s=60,
            marker="x",
            linewidths=1.5,
            label="random",
        )
    ax.scatter(
        xy[picked, 0],
        xy[picked, 1],
        c="red",
        s=70,
        edgecolors="darkred",
        linewidths=0.8,
        label="picked",
        zorder=5,
    )
    ax.set_title(f"Embedding space ({method})")
    ax.legend(loc="best", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def _as_path_list(files) -> list[str] | None:
    if files is None:
        return None
    if isinstance(files, (str, Path)):
        return [str(files)]
    out = []
    for f in files:
        if f is None:
            continue
        out.append(f if isinstance(f, str) else getattr(f, "name", str(f)))
    return out or None


def _zip_path(zip_file) -> str | None:
    if zip_file is None:
        return None
    return zip_file if isinstance(zip_file, str) else getattr(zip_file, "name", None)


def refresh_label_columns(folder_files, zip_file):
    # Peek labels.csv from zip if present; folder uploads already include it among files
    cols: list[str] = []
    paths = _as_path_list(folder_files)
    if paths:
        for p in paths:
            if Path(p).name.lower() == "labels.csv":
                cols = picker.list_label_columns_from_csv(Path(p))
                break
    if not cols:
        zp = _zip_path(zip_file)
        if zp:
            cols = picker.discover_label_columns(zip_path=zp)
    if cols:
        return gr.update(
            choices=cols,
            value=picker.default_label_column(cols),
            interactive=True,
        )
    return gr.update(
        choices=[],
        value=None,
        interactive=False,
        info="No labels.csv — scores use KMeans fallback (still works)",
    )


def run_picker(
    folder_files,
    zip_file,
    prompt: str,
    K: int,
    lam: float,
    quality_percentile: float,
    label_column: str,
    embed_model: str,
    use_umap: bool,
    progress=gr.Progress(track_tqdm=False),
):
    uploaded = _as_path_list(folder_files)
    zip_path = _zip_path(zip_file) if not uploaded else None

    if not uploaded and not zip_path:
        raise gr.Error("Upload a folder of images (or a .zip of that folder)")

    progress(0.05, desc="Loading images…")
    # Run pipeline; wrap steps with progress via a thin hook isn't available,
    # so we mark coarse phases around the call.
    progress(0.15, desc="Embedding images (first run downloads the model)…")

    result = picker.run(
        zip_path=zip_path,
        uploaded_files=uploaded,
        prompt=prompt or "",
        K=int(K),
        lam=float(lam),
        quality_percentile=float(quality_percentile),
        label_column=(label_column or None),
        embed_model=embed_model or "CLIP ViT-B-32",
    )

    progress(0.85, desc="Building galleries…")
    _STATE.clear()
    _STATE.update(result)

    paths = result["paths"]
    picked = result["picked"]
    random_idxs = result["random_idxs"]
    sizes = result["cluster_sizes"]
    n = result["clusters_total"]

    headline = (
        f"*{result['label_source']}*  ·  embed: `{result['embed_model']}`\n\n"
        f"## Clusters covered: {result['clusters_covered']}/{n} "
        f"vs random {result['clusters_covered_random']:.1f}/{n}\n\n"
        f"## Distinct labels in picks: {result['distinct_labels']} "
        f"vs random {result['distinct_labels_random']:.1f}\n\n"
        f"## Missed: {result['pct_missed']:.1f}% "
        f"vs random {result['pct_missed_random']:.1f}%  "
        f"(threshold = {result['miss_threshold']:.3f})"
    )

    outlier_idxs = result.get("outlier_idxs") or []
    fig = _scatter_plot(
        result["embeddings"],
        picked,
        random_idxs,
        outlier_idxs=outlier_idxs,
        use_umap=bool(use_umap),
    )

    picked_gallery = [(paths[i], f"+{sizes[i]} similar") for i in picked]
    random_gallery = [(paths[i], Path(paths[i]).name) for i in random_idxs]

    scores = result["outlier_scores"]
    odd_gallery = [
        (paths[i], f"score {scores[i]:.3f}")
        for i in outlier_idxs
    ]
    odd_title = f"### Odd ones ({result.get('n_outliers', len(outlier_idxs))})"

    details = (
        f"<small>Mean similarity coverage: {result['coverage_picked']:.3f} "
        f"vs random {result['coverage_random']:.3f}  ·  "
        f"outlier threshold (mean 10-NN sim P5) = {result['outlier_threshold']:.3f}</small>\n\n"
        f"Total images: {result['n_total']}  |  "
        f"Kept after quality filter: {result['n_kept']}  |  "
        f"Filtered for low quality: {result['n_filtered']}  |  "
        f"Outliers excluded from picks: {result.get('n_outliers', 0)}"
    )

    labels = [f"{Path(paths[i]).name}  (+{sizes[i]} similar)" for i in picked]
    dropdown = gr.update(choices=labels, value=labels[0] if labels else None)
    first_cluster = result["clusters"][picked[0]] if picked else []
    cluster_gallery = [(p, Path(p).name) for p in first_cluster]

    progress(1.0, desc="Done")
    return (
        headline,
        fig,
        picked_gallery,
        random_gallery,
        odd_title,
        odd_gallery,
        details,
        dropdown,
        cluster_gallery,
    )


def show_cluster(label: str):
    if not _STATE or not label:
        return []
    paths = _STATE["paths"]
    picked = _STATE["picked"]
    sizes = _STATE["cluster_sizes"]
    labels = [f"{Path(paths[i]).name}  (+{sizes[i]} similar)" for i in picked]
    try:
        idx = labels.index(label)
    except ValueError:
        return []
    return [(p, Path(p).name) for p in _STATE["clusters"][picked[idx]]]


def build_ui():
    with gr.Blocks(title="variant-picker") as demo:
        gr.Markdown(
            """
# variant-picker
Upload a folder of ~300 AI image variants. Get the **K** most diverse, high-quality ones —
so a reviewer sees the range, not 40 near-duplicates.

**How to use:** put your images in a folder → upload that folder below → hit **Run**.
No labels needed. First run downloads a model once (~1–2 min on a laptop).
"""
        )

        folder_in = gr.File(
            label="Upload folder of images (png / jpg / webp)",
            file_count="directory",
            file_types=["image", ".csv"],
        )
        zip_in = gr.File(
            label="Or upload a .zip of that folder",
            file_types=[".zip"],
        )

        with gr.Accordion("Advanced (optional)", open=False):
            label_col = gr.Dropdown(
                label="Eval label column (only if your folder/zip includes labels.csv)",
                choices=[],
                value=None,
                interactive=False,
            )
            embed_dd = gr.Dropdown(
                label="Embedding model",
                choices=list(picker.EMBED_MODELS.keys()),
                value="CLIP ViT-B-32",
            )
            prompt_in = gr.Textbox(
                label="Optional prompt (CLIP quality only)",
                placeholder="product photo of a red sneaker on white background",
            )
            use_umap = gr.Checkbox(
                label="Use UMAP for scatter (slower; PCA is default)",
                value=False,
            )
            with gr.Row():
                k_slider = gr.Slider(4, 24, value=12, step=1, label="K (number to pick)")
                lam_slider = gr.Slider(
                    0.0, 2.0, value=0.5, step=0.05, label="lambda (quality weight)"
                )
                q_slider = gr.Slider(
                    0, 40, value=10, step=1, label="Quality percentile to drop"
                )

        # Defaults when accordion closed — still need components for click inputs
        # (they're inside accordion so still registered)

        run_btn = gr.Button("Run", variant="primary")

        headline_out = gr.Markdown()
        scatter_out = gr.Plot(label="2D projection")

        gr.Markdown("### Picked variants")
        picked_gal = gr.Gallery(label="Picked", columns=4, height=360, object_fit="contain")

        gr.Markdown("### Random baseline (same K)")
        random_gal = gr.Gallery(label="Random", columns=4, height=360, object_fit="contain")

        odd_title = gr.Markdown("### Odd ones (0)")
        odd_gal = gr.Gallery(label="Odd ones", columns=4, height=280, object_fit="contain")

        details_out = gr.Markdown()

        gr.Markdown("### Inspect a cluster")
        cluster_dd = gr.Dropdown(label="Picked image", choices=[])
        cluster_gal = gr.Gallery(label="Cluster members", columns=4, height=360, object_fit="contain")

        folder_in.change(fn=refresh_label_columns, inputs=[folder_in, zip_in], outputs=[label_col])
        zip_in.change(fn=refresh_label_columns, inputs=[folder_in, zip_in], outputs=[label_col])

        run_btn.click(
            fn=run_picker,
            inputs=[
                folder_in,
                zip_in,
                prompt_in,
                k_slider,
                lam_slider,
                q_slider,
                label_col,
                embed_dd,
                use_umap,
            ],
            outputs=[
                headline_out,
                scatter_out,
                picked_gal,
                random_gal,
                odd_title,
                odd_gal,
                details_out,
                cluster_dd,
                cluster_gal,
            ],
        )
        cluster_dd.change(fn=show_cluster, inputs=[cluster_dd], outputs=[cluster_gal])

    return demo


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_ui().launch(share=args.share, server_port=args.port)
