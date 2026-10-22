"""Core pipeline: load → embed → quality → select → cluster → metrics."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_SIDE = 256
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
DINO_MODEL_ID = "facebook/dinov2-small"

EMBED_MODELS = {
    "CLIP ViT-B-32": "clip",
    "DINOv2-small": "dinov2",
}

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_dino_model = None
_dino_processor = None
_device = None


def _log(step: str, t0: float) -> None:
    print(f"[{step}] {time.perf_counter() - t0:.1f}s")


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cpu")
    return _device


def _get_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        t0 = time.perf_counter()
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        _clip_model = _clip_model.to(_get_device()).eval()
        _clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        _log("load_clip_model", t0)
    return _clip_model, _clip_preprocess, _clip_tokenizer


def _get_dino():
    global _dino_model, _dino_processor
    if _dino_model is None:
        t0 = time.perf_counter()
        from transformers import AutoImageProcessor, AutoModel

        _dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
        _dino_model = AutoModel.from_pretrained(DINO_MODEL_ID).to(_get_device()).eval()
        _log("load_dinov2_model", t0)
    return _dino_model, _dino_processor


def _resize_for_speed(img: Image.Image) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= MAX_SIDE:
        return img
    scale = MAX_SIDE / m
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)


def _folder_cache_key(folder: Path, rel_paths: list[str], model_key: str) -> str:
    h = hashlib.sha256()
    h.update(str(folder.resolve()).encode())
    h.update(model_key.encode())
    h.update("\n".join(rel_paths).encode())
    return h.hexdigest()[:16]


def _rel_to(path: str | Path, root: Path) -> str:
    """Relative path that survives macOS /var vs /private/var resolution."""
    return str(Path(path).resolve().relative_to(root.resolve()))


def _resolve_model_key(embed_model: str) -> str:
    if embed_model in EMBED_MODELS:
        return EMBED_MODELS[embed_model]
    if embed_model in ("clip", "dinov2"):
        return embed_model
    return "clip"


def find_labels_csv(root: Path) -> Path | None:
    csv_path = root / "labels.csv"
    if csv_path.is_file():
        return csv_path
    found = list(root.rglob("labels.csv"))
    return found[0] if found else None


_FILENAME_COLS = {"filename", "file", "path", "filepath", "image", "img"}
# Prefer meaningful eval columns; avoid per-image IDs like seed (always unique → trivial scores).
_PREFERRED_LABEL_COLS = ("colour", "color", "cluster", "label", "class", "category")


def list_label_columns_from_csv(csv_path: Path) -> list[str]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        cols = [c for c in reader.fieldnames if c and c.lower() not in _FILENAME_COLS]
        return _order_label_columns(cols)


def _order_label_columns(cols: list[str]) -> list[str]:
    """Put preferred eval columns first; push seed/id-like columns last."""
    lower = {c.lower(): c for c in cols}
    ordered: list[str] = []
    for pref in _PREFERRED_LABEL_COLS:
        if pref in lower:
            ordered.append(lower[pref])
    weak = {"seed", "id", "index", "idx", "name"}
    rest = [c for c in cols if c not in ordered and c.lower() not in weak]
    weak_present = [c for c in cols if c.lower() in weak and c not in ordered]
    return ordered + rest + weak_present


def default_label_column(cols: list[str]) -> str | None:
    if not cols:
        return None
    return _order_label_columns(cols)[0]


def discover_label_columns(zip_path: str | None = None, folder: str | None = None) -> list[str]:
    """Return non-filename columns from labels.csv for the UI dropdown."""
    if folder:
        root = Path(folder).expanduser().resolve()
        if root.is_dir():
            csv_path = find_labels_csv(root)
            if csv_path:
                return list_label_columns_from_csv(csv_path)
    if zip_path:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = [n for n in zf.namelist() if n.endswith("labels.csv") and not n.endswith("/")]
                if not names:
                    return []
                raw = zf.read(names[0])
                reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
                if not reader.fieldnames:
                    return []
                cols = [c for c in reader.fieldnames if c and c.lower() not in _FILENAME_COLS]
                return _order_label_columns(cols)
        except Exception as e:
            print(f"  discover_label_columns zip failed: {e}")
    return []


def _stage_uploaded_files(uploaded_files: list) -> Path:
    """Copy uploaded files (folder or multi-file) into a fresh temp root."""
    import shutil

    root = Path(tempfile.mkdtemp(prefix="variant_picker_")).resolve()
    for item in uploaded_files:
        src = Path(item if isinstance(item, str) else getattr(item, "name", str(item)))
        if not src.is_file():
            continue
        # Preserve basename; uniquify collisions
        dest = root / src.name
        if dest.exists():
            dest = root / f"{src.stem}_{abs(hash(str(src))) % 10_0000}{src.suffix}"
        shutil.copy2(src, dest)
    return root


def _collect_images(root: Path) -> tuple[list[str], list[Image.Image], int]:
    paths: list[str] = []
    images: list[Image.Image] = []
    skipped = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            img = Image.open(p).convert("RGB")
            images.append(_resize_for_speed(img))
            paths.append(str(p.resolve()))
        except Exception as e:
            skipped += 1
            print(f"  skip corrupted: {p.name} ({e})")
    return paths, images, skipped


def load_images(
    zip_path: str | None = None,
    folder: str | None = None,
    uploaded_files: list | None = None,
) -> tuple[list[str], list[Image.Image], Path]:
    """Load images from a zip, local folder, or uploaded file list. Labels.csv optional."""
    t0 = time.perf_counter()

    if uploaded_files:
        root = _stage_uploaded_files(uploaded_files)
    elif zip_path:
        extract_dir = Path(tempfile.mkdtemp(prefix="variant_picker_")).resolve()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        root = extract_dir
    elif folder:
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Folder not found: {root}")
    else:
        raise ValueError("Upload a folder of images (or a zip)")

    paths, images, skipped = _collect_images(root)
    if not paths:
        raise ValueError("No valid images found (png / jpg / webp)")

    _log(f"load_images ({len(paths)} ok, {skipped} skipped)", t0)
    return paths, images, root


def _embed_clip(images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
    model, preprocess, _ = _get_clip()
    device = _get_device()
    all_emb: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([preprocess(im) for im in batch]).to(device)
            feats = model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_emb.append(feats.cpu().numpy().astype(np.float32))
    return np.concatenate(all_emb, axis=0)


def _embed_dinov2(images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
    model, processor = _get_dino()
    device = _get_device()
    all_emb: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            # CLS token
            feats = out.last_hidden_state[:, 0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_emb.append(feats.cpu().numpy().astype(np.float32))
    return np.concatenate(all_emb, axis=0)


def embed_images(
    paths: list[str],
    images: list[Image.Image],
    root: Path,
    embed_model: str = "clip",
    batch_size: int = 32,
) -> np.ndarray:
    """Embed images, L2-normalize, cache to .npy keyed by folder+file list+model."""
    t0 = time.perf_counter()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    model_key = _resolve_model_key(embed_model)

    rel_paths = sorted(_rel_to(p, root) for p in paths)
    key = _folder_cache_key(root, rel_paths, model_key)
    npy_path = CACHE_DIR / f"{model_key}_{key}.npy"
    meta_path = CACHE_DIR / f"{model_key}_{key}.json"

    if npy_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("rel_paths") == rel_paths and meta.get("model") == model_key:
            emb = np.load(npy_path)
            _log(f"embed_images (cache hit {model_key} {emb.shape})", t0)
            return emb

    if model_key == "dinov2":
        emb = _embed_dinov2(images, batch_size=min(batch_size, 16))
    else:
        emb = _embed_clip(images, batch_size=batch_size)

    np.save(npy_path, emb)
    meta_path.write_text(
        json.dumps({"rel_paths": [_rel_to(p, root) for p in paths], "model": model_key})
    )
    _log(f"embed_images ({model_key} {emb.shape})", t0)
    return emb


def quality_scores(
    images: list[Image.Image],
    embeddings: np.ndarray,
    prompt: str = "",
    embed_model: str = "clip",
) -> np.ndarray:
    """Sharpness + optional CLIP prompt match, averaged and min-max normalized to [0, 1]."""
    t0 = time.perf_counter()
    n = len(images)
    sharp = np.zeros(n, dtype=np.float32)

    for i, img in enumerate(images):
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        sharp[i] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _minmax(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-12:
            return np.ones_like(x)
        return (x - lo) / (hi - lo)

    components = [_minmax(sharp)]
    model_key = _resolve_model_key(embed_model)

    if prompt and prompt.strip() and model_key == "clip":
        model, _, tokenizer = _get_clip()
        device = _get_device()
        with torch.no_grad():
            tokens = tokenizer([prompt.strip()]).to(device)
            text_feat = model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            text_np = text_feat.cpu().numpy().astype(np.float32)[0]
        prompt_sim = embeddings @ text_np
        components.append(_minmax(prompt_sim.astype(np.float32)))
    elif prompt and prompt.strip() and model_key != "clip":
        print("  prompt match skipped (only available with CLIP embeddings)")

    quality = np.mean(np.stack(components, axis=0), axis=0).astype(np.float32)
    quality = _minmax(quality)
    _log("quality_scores", t0)
    return quality


def filter_by_quality(
    quality: np.ndarray,
    percentile: float = 10.0,
) -> np.ndarray:
    """Return boolean mask of images to keep (above percentile threshold)."""
    if percentile <= 0:
        return np.ones(len(quality), dtype=bool)
    thresh = float(np.percentile(quality, percentile))
    mask = quality >= thresh
    if mask.sum() == 0:
        mask[:] = True
    return mask


def mean_knn_similarity(S: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-image mean cosine similarity to its k nearest neighbors (self excluded)."""
    n = S.shape[0]
    if n < 2:
        return np.ones(n, dtype=np.float64)
    k = min(k, n - 1)
    S_nn = S.copy()
    np.fill_diagonal(S_nn, -np.inf)
    # largest k similarities per row
    partial = np.partition(S_nn, -k, axis=1)[:, -k:]
    return partial.mean(axis=1)


def flag_outliers(
    S: np.ndarray,
    k_nn: int = 10,
    percentile: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Outlier if mean similarity to k_nn nearest neighbors is below the given percentile.
    Returns (is_outlier mask, mean_knn scores, threshold). Lower score = more unusual.
    """
    t0 = time.perf_counter()
    scores = mean_knn_similarity(S, k=k_nn)
    if len(scores) < 3:
        mask = np.zeros(len(scores), dtype=bool)
        thresh = float(scores.min()) if len(scores) else 0.0
        _log("flag_outliers (skipped, too few)", t0)
        return mask, scores, thresh
    thresh = float(np.percentile(scores, percentile))
    mask = scores < thresh
    _log(f"flag_outliers ({int(mask.sum())} below P{percentile:g}={thresh:.3f})", t0)
    return mask, scores, thresh


def greedy_select(
    S: np.ndarray,
    quality: np.ndarray,
    K: int,
    lam: float = 0.5,
    candidates: list[int] | None = None,
) -> list[int]:
    """Greedy facility-location with quality weight. Optional candidate index list."""
    t0 = time.perf_counter()
    n = S.shape[0]
    if candidates is None:
        candidates = list(range(n))
    else:
        candidates = list(candidates)
    if not candidates:
        raise ValueError("No candidates left for selection (all flagged as outliers?)")
    K = min(K, len(candidates))
    coverage = np.zeros(n, dtype=np.float64)
    picked: list[int] = []
    remaining = set(candidates)

    for _ in range(K):
        best_c = -1
        best_gain = -np.inf
        for c in remaining:
            gain = float(np.maximum(0.0, S[:, c] - coverage).sum()) + lam * float(quality[c])
            if gain > best_gain:
                best_gain = gain
                best_c = c
        picked.append(best_c)
        remaining.remove(best_c)
        coverage = np.maximum(coverage, S[:, best_c])

    _log(f"greedy_select (K={K})", t0)
    return picked


def coverage_metric(S: np.ndarray, picked: list[int]) -> float:
    if not picked:
        return 0.0
    return float(S[:, picked].max(axis=1).mean())


def nearest_neighbor_sims(S: np.ndarray) -> np.ndarray:
    """Per-image max cosine similarity to any other image (self excluded)."""
    n = S.shape[0]
    if n < 2:
        return np.ones(n, dtype=np.float64)
    S_nn = S.copy()
    np.fill_diagonal(S_nn, -np.inf)
    return S_nn.max(axis=1)


def data_relative_miss_threshold(S: np.ndarray) -> float:
    """Median of nearest-neighbor cosine similarities."""
    nn = nearest_neighbor_sims(S)
    return float(np.median(nn))


def pct_missed(S: np.ndarray, picked: list[int], threshold: float) -> float:
    """Percent of images whose max cosine sim to any pick is below threshold."""
    if not picked:
        return 100.0
    max_sim = S[:, picked].max(axis=1)
    return float(100.0 * (max_sim < threshold).mean())


def clusters_covered(cluster_ids: np.ndarray, picked: list[int]) -> tuple[int, int]:
    """How many unique clusters contain at least one picked image. Returns (covered, total)."""
    n_clusters = int(len(set(int(x) for x in cluster_ids.tolist())))
    if not picked or n_clusters == 0:
        return 0, n_clusters
    covered = len({int(cluster_ids[i]) for i in picked})
    return covered, n_clusters


def distinct_labels_in_picks(cluster_ids: np.ndarray, picked: list[int]) -> int:
    if not picked:
        return 0
    return len({int(cluster_ids[i]) for i in picked})


def load_source_labels(
    root: Path,
    paths: list[str],
    label_column: str | None,
) -> tuple[np.ndarray | None, str | None]:
    """Load integer cluster ids from labels.csv column. Returns (ids, column_name) or (None, None)."""
    csv_path = find_labels_csv(root)
    if csv_path is None:
        return None, None

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None, None
        fields = list(reader.fieldnames)
        fname_col = None
        for cand in ("filename", "file", "path", "filepath", "image", "img"):
            for name in fields:
                if name and name.lower() == cand:
                    fname_col = name
                    break
            if fname_col:
                break
        if fname_col is None:
            print("  labels.csv has no filename column; ignoring")
            return None, None

        skip = {fname_col.lower()}
        cols = _order_label_columns([c for c in fields if c and c.lower() not in skip])
        if not cols:
            print("  labels.csv has no label columns; ignoring")
            return None, None

        col = label_column if label_column in cols else default_label_column(cols)
        mapping: dict[str, str] = {}
        for row in reader:
            mapping[Path(row[fname_col]).name] = row[col]

    labels_str = []
    for p in paths:
        name = Path(p).name
        if name not in mapping:
            print(f"  labels.csv incomplete (missing {name}); falling back to KMeans")
            return None, None
        labels_str.append(mapping[name])

    uniq = sorted(set(labels_str))
    to_id = {lab: i for i, lab in enumerate(uniq)}
    ids = np.array([to_id[s] for s in labels_str], dtype=np.int32)
    print(f"  loaded labels.csv column={col!r} ({len(uniq)} values)")
    return ids, col


def kmeans_cluster(embeddings: np.ndarray) -> tuple[np.ndarray, int]:
    """KMeans with k by silhouette (5..20). Returns (ids, k)."""
    t0 = time.perf_counter()
    n = len(embeddings)
    k_min, k_max = 5, 20
    if n <= k_min:
        k = max(2, n - 1) if n > 2 else 1
        if k <= 1:
            _log("cluster_embeddings (trivial)", t0)
            return np.zeros(n, dtype=np.int32), 1
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        ids = km.fit_predict(embeddings).astype(np.int32)
        _log(f"cluster_embeddings (KMeans k={k})", t0)
        return ids, k

    k_max = min(k_max, n - 1)
    best_k, best_score, best_ids = k_min, -np.inf, None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        ids = km.fit_predict(embeddings)
        score = float(silhouette_score(embeddings, ids, metric="cosine"))
        if score > best_score:
            best_k, best_score, best_ids = k, score, ids
    _log(f"cluster_embeddings (KMeans k={best_k}, sil={best_score:.3f})", t0)
    return best_ids.astype(np.int32), best_k


def assign_cluster_ids(
    embeddings: np.ndarray,
    paths: list[str],
    root: Path,
    label_column: str | None,
) -> tuple[np.ndarray, str]:
    """
    Prefer labels.csv column; else KMeans.
    Returns (ids, source_description) e.g. 'labels: colour' or 'labels: KMeans k=N (fallback)'.
    """
    ids, col = load_source_labels(root, paths, label_column)
    if ids is not None and col is not None:
        return ids, f"labels: {col}"
    ids, k = kmeans_cluster(embeddings)
    return ids, f"labels: KMeans k={k} (fallback)"


def random_baselines(
    S: np.ndarray,
    cluster_ids: np.ndarray,
    K: int,
    miss_threshold: float,
    n_trials: int = 50,
    candidate_pool: list[int] | None = None,
) -> tuple[float, float, float, float]:
    """Average (clusters_covered, distinct_labels, pct_missed, mean_sim) over random K-subsets."""
    t0 = time.perf_counter()
    pool = list(range(S.shape[0])) if candidate_pool is None else list(candidate_pool)
    K = min(K, len(pool))
    cov_counts, distincts, missed, means = [], [], [], []
    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        idxs = rng.choice(pool, size=K, replace=False).tolist()
        covered, _ = clusters_covered(cluster_ids, idxs)
        cov_counts.append(covered)
        distincts.append(distinct_labels_in_picks(cluster_ids, idxs))
        missed.append(pct_missed(S, idxs, miss_threshold))
        means.append(coverage_metric(S, idxs))
    _log(f"random_baselines ({n_trials} trials)", t0)
    return (
        float(np.mean(cov_counts)),
        float(np.mean(distincts)),
        float(np.mean(missed)),
        float(np.mean(means)),
    )


def build_clusters(
    S: np.ndarray,
    picked: list[int],
    paths: list[str],
) -> dict[int, list[str]]:
    """Assign each non-picked image to its most similar picked image. Includes the pick itself."""
    t0 = time.perf_counter()
    clusters: dict[int, list[str]] = {p: [paths[p]] for p in picked}
    picked_set = set(picked)
    for i in range(S.shape[0]):
        if i in picked_set:
            continue
        best = max(picked, key=lambda p: S[i, p])
        clusters[best].append(paths[i])
    _log("build_clusters", t0)
    return clusters


def run(
    zip_path: str | None = None,
    folder: str | None = None,
    uploaded_files: list | None = None,
    prompt: str = "",
    K: int = 12,
    lam: float = 0.5,
    quality_percentile: float = 10.0,
    label_column: str | None = None,
    embed_model: str = "CLIP ViT-B-32",
) -> dict:
    """Full pipeline. Returns a result dict for the Gradio UI."""
    paths, images, root = load_images(
        zip_path=zip_path, folder=folder, uploaded_files=uploaded_files
    )
    n_total = len(paths)
    model_key = _resolve_model_key(embed_model)

    embeddings = embed_images(paths, images, root, embed_model=model_key)
    quality = quality_scores(images, embeddings, prompt=prompt, embed_model=model_key)
    keep = filter_by_quality(quality, percentile=quality_percentile)
    n_filtered = int((~keep).sum())

    paths_f = [p for p, k in zip(paths, keep) if k]
    emb_f = embeddings[keep]
    qual_f = quality[keep]

    if len(paths_f) < 1:
        raise ValueError("All images filtered out; lower the quality percentile")

    S = emb_f @ emb_f.T
    np.clip(S, -1.0, 1.0, out=S)

    # Flag outliers before selection; exclude them from the main K picks.
    outlier_mask, outlier_scores, outlier_thresh = flag_outliers(S, k_nn=10, percentile=5.0)
    inlier_idxs = np.where(~outlier_mask)[0].tolist()
    if not inlier_idxs:
        # Degenerate: treat everyone as inlier
        inlier_idxs = list(range(len(paths_f)))
        outlier_mask[:] = False

    K = min(int(K), len(inlier_idxs))

    # Selection uses ONLY similarity + quality — never labels.
    miss_threshold = data_relative_miss_threshold(S)
    picked = greedy_select(S, qual_f, K=K, lam=lam, candidates=inlier_idxs)

    # Labels are evaluation-only (score the picks afterward).
    cluster_ids, label_source = assign_cluster_ids(emb_f, paths_f, root, label_column)
    covered, n_clusters = clusters_covered(cluster_ids, picked)
    distinct = distinct_labels_in_picks(cluster_ids, picked)
    missed = pct_missed(S, picked, threshold=miss_threshold)
    cov_picked = coverage_metric(S, picked)

    # Random baseline also drawn from inliers only (same pool as picks).
    rand_covered, rand_distinct, rand_missed, cov_random = random_baselines(
        S,
        cluster_ids,
        K=K,
        miss_threshold=miss_threshold,
        n_trials=50,
        candidate_pool=inlier_idxs,
    )

    rng = np.random.default_rng(0)
    random_idxs = rng.choice(inlier_idxs, size=K, replace=False).tolist()

    assign = build_clusters(S, picked, paths_f)
    cluster_sizes = {p: len(assign[p]) - 1 for p in picked}

    # Odd ones: outliers sorted most unusual first (lowest mean-kNN similarity).
    outlier_idxs = np.where(outlier_mask)[0]
    outlier_order = outlier_idxs[np.argsort(outlier_scores[outlier_idxs])]

    return {
        "paths": paths_f,
        "embeddings": emb_f,
        "picked": picked,
        "random_idxs": random_idxs,
        "clusters": assign,
        "cluster_sizes": cluster_sizes,
        "cluster_ids": cluster_ids,
        "label_source": label_source,
        "clusters_covered": covered,
        "clusters_total": n_clusters,
        "clusters_covered_random": rand_covered,
        "distinct_labels": distinct,
        "distinct_labels_random": rand_distinct,
        "pct_missed": missed,
        "pct_missed_random": rand_missed,
        "coverage_picked": cov_picked,
        "coverage_random": cov_random,
        "n_total": n_total,
        "n_filtered": n_filtered,
        "n_kept": len(paths_f),
        "quality": qual_f,
        "miss_threshold": miss_threshold,
        "embed_model": model_key,
        "outlier_mask": outlier_mask,
        "outlier_scores": outlier_scores,
        "outlier_threshold": outlier_thresh,
        "outlier_idxs": outlier_order.tolist(),
        "n_outliers": int(outlier_mask.sum()),
    }
