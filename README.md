# variant-picker

Pick the K most diverse, high-quality image variants from a large AI-generated set so a reviewer sees the full range of options instead of near-duplicates.

## Setup

```bash
cd variant-picker
python3 -m venv .venv   # 3.11+ recommended
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k` weights once (~350 MB).

## Quick test (no real campaign images)

If you have ~10 seed images in a folder:

```bash
python make_test_set.py --src /path/to/seeds --out ./test_variants --n 300
python app.py
```

In the UI, set the folder path to `./test_variants` (or the absolute path) and hit Run.

## Run

```bash
python app.py
# public link for cold emails / demos:
python app.py --share
```

Then open the Gradio URL in your browser.

### How testers use it

1. Put ~300 images in a folder (png / jpg / webp)
2. Upload that folder in the UI (or zip it and upload the zip)
3. Hit **Run**

No `labels.csv` required — scoring falls back to KMeans. If you include a `labels.csv`, pick a column like `colour` or `cluster` under Advanced (not `seed`).

### Inputs

- **Folder upload** (primary) or **zip** of images
- Optional **prompt** for CLIP prompt-match quality (Advanced)
- **K**, **lambda**, quality percentile, embedding model (Advanced)

### Outputs

1. Headline metrics vs random baseline (50 subsets): clusters covered, distinct labels in picks, % missed (data-relative threshold = median nearest-neighbor cosine)
2. Label source shown as `labels: <column>` or `labels: KMeans k=N (fallback)`
3. 2D scatter of embeddings (UMAP if installed, else PCA)
4. Gallery of K picked images (captioned with cluster size)
5. Gallery of K random images for comparison
6. Mean-similarity coverage (small text) plus image counts
7. Dropdown to inspect a picked image's cluster

Embeddings are cached under `.cache/` per model so re-runs are fast.
