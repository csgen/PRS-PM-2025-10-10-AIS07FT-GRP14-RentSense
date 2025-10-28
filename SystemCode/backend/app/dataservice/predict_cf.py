"""
Predict top-K property recommendations for a given device_id
using the trained ALS model (BM25 + alpha version).

Usage:
python SystemCode/backend/app/dataservice/predict_cf.py --user-id device_1 --k 10
"""

import os
import json
import argparse
import numpy as np
from joblib import load

# === Paths ===
MODEL_DIR = os.path.join("SystemCode", "backend", "app", "dataservice", "models")
ALS_PATH = os.path.join(MODEL_DIR, "als_model.pkl")
U2I_PATH = os.path.join(MODEL_DIR, "user2idx.json")
I2I_PATH = os.path.join(MODEL_DIR, "item2idx.json")

# === Load model ===
if not (os.path.exists(ALS_PATH) and os.path.exists(U2I_PATH) and os.path.exists(I2I_PATH)):
    raise FileNotFoundError("❌ Model files not found. Please train the model first.")

print(f"[INFO] Loading ALS model and mappings from {MODEL_DIR} ...")
model = load(ALS_PATH)
user2idx = json.load(open(U2I_PATH, encoding="utf-8"))
item2idx = json.load(open(I2I_PATH, encoding="utf-8"))
idx2item = {v: k for k, v in item2idx.items()}

# === Parse CLI args ===
parser = argparse.ArgumentParser()
parser.add_argument("--user-id", required=True, help="User ID (e.g. device_1)")
parser.add_argument("--k", type=int, default=10, help="Top-K recommendations")
args = parser.parse_args()

user_id = args.user_id
TOP_K = args.k

# === Recommend ===
if user_id not in user2idx:
    # Cold-start fallback: just return top K popular items (IDs only)
    print({
        "user_id": user_id,
        "cold_start": True,
        "recommendations": list(item2idx.keys())[:TOP_K]
    })
else:
    u = user2idx[user_id]
    try:
        item_idx, scores = model.recommend(
            userid=u,
            user_items=None,                   # 无需用户历史矩阵
            N=TOP_K,
            filter_already_liked_items=False,   # 若要过滤训练集中看过的房源，可设 True
            recalculate_user=False              # 防止 posv 数值报错
        )
    except Exception as e:
        print(f"⚠️ Recommendation failed: {e}")
        exit(1)

    # --- Normalize scores to [0,1] for display ---
    scores = np.array(scores, dtype=float)
    if len(scores) > 0:
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    else:
        norm_scores = scores

    recs = [
        {"property_id": idx2item[i], "score_raw": float(s), "score_norm": float(sn)}
        for i, s, sn in zip(item_idx, scores, norm_scores)
    ]

    print({
        "user_id": user_id,
        "topk": TOP_K,
        "cold_start": False,
        "recommendations": recs
    })
