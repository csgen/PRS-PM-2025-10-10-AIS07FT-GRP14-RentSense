"""
Offline training script for Collaborative Filtering (ALS)
with BM25 weighting + alpha scaling for stronger implicit signal.

It also exports item popularity metrics for better cold-start fallback:
- item_popularity.json  (sum of implicit strengths per item)
- item_pop_count.json   (count of interactions per item)
- global_stats.json     (global means, sizes, top popular IDs)

Usage example (Windows PowerShell):
python SystemCode/backend/app/dataservice/DataScript/train_cf.py `
  --input-path "Miscellaneous/behaviors.csv" `
  --output-dir "SystemCode/backend/app/dataservice/models" `
  --model-type als `
  --user-col device_id `
  --item-col property_id `
  --rating-mode implicit `
  --dwell-col dwell_time `
  --favorite-col favorite `
  --rank 64 --regularization 0.2 --epochs 20 --k 10 `
  --alpha 40 --fav-weight 3.0
"""

import argparse
import json
import logging
import os
from typing import Dict, Tuple

from joblib import dump
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------- Utilities ----------------
def _parse_favorite(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    if series.dtype == object:
        s = series.astype(str).str.strip().str.upper()
        mapping = {"TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "YES": 1, "NO": 0}
        return s.map(mapping).fillna(0).astype(int)
    return series.fillna(0).astype(float).clip(lower=0, upper=1).round().astype(int)


def load_interactions(path, user_col, item_col, dwell_col=None, favorite_col=None,
                      rating_mode="implicit", fav_weight=1.0) -> pd.DataFrame:
    """
    Load and construct implicit strength:
        strength = normalized(dwell) + fav_weight * favorite
    """
    df = pd.read_csv(path, encoding="utf-8-sig")  # 兼容带/不带 BOM
    keep_cols = [c for c in [user_col, item_col, dwell_col, favorite_col] if c and c in df.columns]
    if keep_cols:
        df = df[keep_cols].copy()

    if rating_mode == "implicit":
        dwell = df.get(dwell_col, 0)
        dwell = pd.to_numeric(dwell, errors="coerce").fillna(0.0).astype(float)
        m = float(dwell.max()) if len(dwell) else 0.0
        dwell_norm = (dwell / m) if m > 0 else pd.Series(0.0, index=df.index)
        strength = dwell_norm
        if favorite_col in df.columns:
            fav = _parse_favorite(df[favorite_col])
            strength = strength + float(fav_weight) * fav
        df["strength"] = strength.astype(float)
    else:
        if "rating" not in df.columns:
            raise ValueError("Explicit rating_mode requires a 'rating' column.")
        df["strength"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0).astype(float)

    df = df.rename(columns={user_col: "user_id", item_col: "item_id"})
    df = df[["user_id", "item_id", "strength"]].dropna()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df = df[df["strength"] > 0]
    return df


def precision_at_k(model, train_mtx, test_mtx, K=10) -> float:
    """Mean Precision@K, skip empty or erroneous users."""
    n_users = train_mtx.shape[0]
    precisions = []
    skipped_empty = skipped_err = 0

    train_mtx = train_mtx.tocsr()
    test_mtx = test_mtx.tocsr()

    for u in range(n_users):
        if train_mtx[u].nnz == 0:
            skipped_empty += 1
            continue
        test_items = test_mtx[u].indices
        if test_items.size == 0:
            continue
        try:
            item_ids, _scores = model.recommend(
                userid=u,
                user_items=train_mtx[u],
                N=K,
                filter_already_liked_items=True,
                recalculate_user=False,
            )
        except Exception:
            skipped_err += 1
            continue
        if len(item_ids) == 0:
            continue
        hit = len(set(item_ids[:K]) & set(test_items))
        precisions.append(hit / float(K))
    if skipped_empty:
        logger.info(f"precision@K: skipped {skipped_empty} users with empty train rows.")
    if skipped_err:
        logger.info(f"precision@K: skipped {skipped_err} users due to numerical issues.")
    return float(np.mean(precisions)) if precisions else 0.0


def build_csr(df: pd.DataFrame) -> Tuple[coo_matrix, Dict[str, int], Dict[str, int]]:
    """Build user-item CSR and index maps from dataframe."""
    users = df["user_id"].astype(str).unique()
    items = df["item_id"].astype(str).unique()
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}

    rows = df["user_id"].map(user2idx).to_numpy()
    cols = df["item_id"].map(item2idx).to_numpy()
    data = df["strength"].astype(np.float32).to_numpy()
    ui = coo_matrix((data, (rows, cols)), shape=(len(user2idx), len(item2idx))).tocsr()
    return ui, user2idx, item2idx


def train_als(df: pd.DataFrame, rank, regularization, epochs, k, alpha=40.0):
    """Train ALS with BM25 weighting + alpha scaling."""
    ui, user2idx, item2idx = build_csr(df)

    # Train/test split on interactions
    rng = np.random.default_rng(42)
    coo_m = ui.tocoo()
    mask = rng.random(coo_m.nnz) < 0.2
    ui_train = coo_matrix(
        (coo_m.data[~mask], (coo_m.row[~mask], coo_m.col[~mask])),
        shape=ui.shape
    ).tocsr()
    ui_test = coo_matrix(
        (coo_m.data[mask], (coo_m.row[mask], coo_m.col[mask])),
        shape=ui.shape
    ).tocsr()

    # BM25 + alpha
    logger.info(f"Applying BM25 weighting + alpha={alpha}")
    ui_train = bm25_weight(ui_train, K1=100, B=0.8)
    ui_train = (ui_train * alpha).tocsr()
    ui_test = bm25_weight(ui_test, K1=100, B=0.8)
    ui_test = (ui_test * alpha).tocsr()

    # Train ALS
    model = AlternatingLeastSquares(
        factors=rank,
        regularization=regularization,
        iterations=epochs,
        num_threads=0,
        random_state=42,
    )
    model.fit(ui_train.T.tocsr())

    # Evaluate
    prec = precision_at_k(model, ui_train, ui_test, K=k)
    metrics = {"precision_at_k": float(prec)}
    return model, user2idx, item2idx, metrics


def export_popularity_artifacts(df: pd.DataFrame, out_dir: str, topn_preview: int = 50) -> None:
    """
    Export popularity artifacts used by predict stage for cold-start fallback.
    - item_popularity.json : sum of strengths per item
    - item_pop_count.json  : interaction counts per item
    - global_stats.json    : global means and top popular ids (preview)
    """
    pop_sum = df.groupby("item_id")["strength"].sum().sort_values(ascending=False)
    pop_cnt = df.groupby("item_id")["strength"].size().sort_values(ascending=False)

    pop_sum_dict = {str(k): float(v) for k, v in pop_sum.items()}
    pop_cnt_dict = {str(k): int(v) for k, v in pop_cnt.items()}

    top_pop_items = list(pop_sum.index[:topn_preview])

    global_stats = {
        "n_rows": int(len(df)),
        "n_users": int(df["user_id"].nunique()),
        "n_items": int(df["item_id"].nunique()),
        "global_strength_mean": float(df["strength"].mean()) if len(df) else 0.0,
        "top_pop_items_preview": top_pop_items,
        "pop_metric": "sum_strength",
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "item_popularity.json"), "w", encoding="utf-8") as f:
        json.dump(pop_sum_dict, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "item_pop_count.json"), "w", encoding="utf-8") as f:
        json.dump(pop_cnt_dict, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "global_stats.json"), "w", encoding="utf-8") as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=2)

    logger.info(f"Exported popularity artifacts to {out_dir} "
                f"(top {topn_preview} preview: {top_pop_items[:10]} ...)")


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", default="als")
    parser.add_argument("--user-col", default="device_id")
    parser.add_argument("--item-col", default="property_id")
    parser.add_argument("--rating-mode", default="implicit")
    parser.add_argument("--dwell-col", default="dwell_time")
    parser.add_argument("--favorite-col", default="favorite")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--regularization", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument("--fav-weight", type=float, default=3.0)
    args = parser.parse_args()

    # 1) Load data
    df = load_interactions(
        args.input_path,
        args.user_col,
        args.item_col,
        args.dwell_col,
        args.favorite_col,
        args.rating_mode,
        fav_weight=args.fav_weight,
    )
    logger.info(
        f"Loaded implicit interactions: {len(df)} rows, "
        f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items"
    )

    # 2) Train ALS
    model, user2idx, item2idx, metrics = train_als(
        df, args.rank, args.regularization, args.epochs, args.k, alpha=args.alpha
    )

    # 3) Save model & mappings
    os.makedirs(args.output_dir, exist_ok=True)
    dump(model, os.path.join(args.output_dir, "als_model.pkl"))
    with open(os.path.join(args.output_dir, "user2idx.json"), "w", encoding="utf-8") as f:
        json.dump(user2idx, f, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "item2idx.json"), "w", encoding="utf-8") as f:
        json.dump(item2idx, f, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info(f"Model saved to {args.output_dir}")
    logger.info(f"Metrics: {metrics}")

    # 4) Export popularity artifacts for cold-start fallback (predict stage)
    export_popularity_artifacts(df, args.output_dir, topn_preview=50)


if __name__ == "__main__":
    main()
