"""
Predict top-K property recommendations for a given device_id
using the trained ALS model.

Supports:
- Global recommend (no candidates)
- Candidate ranking (with fallback for unseen items via content scores / mean/zero/constant)
- Cold-start user:
    - with candidates -> rank by content scores (or fallback)
    - without candidates -> use popularity fallback (item_popularity.json)

Usage examples:
python SystemCode/backend/app/dataservice/predict_cf.py --user-id device_1 --k 10
python SystemCode/backend/app/dataservice/predict_cf.py --user-id device_1 --k 50 --candidates 309,311,999999,888888 --content-scores-file content_scores.json
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
POP_PATH = os.path.join(MODEL_DIR, "item_popularity.json")

# === Sanity ===
if not (os.path.exists(ALS_PATH) and os.path.exists(U2I_PATH) and os.path.exists(I2I_PATH)):
    raise FileNotFoundError("❌ Model files not found. Please train the model first.")

print(f"[INFO] Loading ALS model and mappings from {MODEL_DIR} ...")
model = load(ALS_PATH)
user2idx = json.load(open(U2I_PATH, encoding="utf-8-sig"))
item2idx = json.load(open(I2I_PATH, encoding="utf-8-sig"))
idx2item = {v: k for k, v in item2idx.items()}

# Optional popularity for cold-start no-candidate fallback
item_popularity = {}
if os.path.exists(POP_PATH):
    try:
        item_popularity = json.load(open(POP_PATH, encoding="utf-8-sig"))
    except Exception:
        item_popularity = {}

# === Helpers ===
def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.ones_like(arr, dtype=float) * 0.5
    return (arr - lo) / (hi - lo + 1e-12)

def _load_candidates_from_args(args) -> list[str] | None:
    # 1) inline comma list
    if args.candidates:
        return [s.strip() for s in args.candidates.split(",") if s.strip()]
    # 2) from file (csv/json)
    if args.candidates_file:
        path = args.candidates_file
        if path.lower().endswith(".json"):
            data = json.load(open(path, encoding="utf-8-sig"))
            if isinstance(data, list):
                if data and isinstance(data[0], dict) and "property_id" in data[0]:
                    return [str(d["property_id"]) for d in data]
                return [str(x) for x in data]
        else:
            # CSV with BOM tolerance
            with open(path, "r", encoding="utf-8-sig") as f:
                header = f.readline().strip().split(",")
                header = [h.lstrip("\ufeff") for h in header]
                rows = [line.strip().split(",") for line in f if line.strip()]
            if len(header) == 1:
                return [r[0] for r in rows]
            if "property_id" in header:
                idx = header.index("property_id")
                return [r[idx] for r in rows]
            return [r[0] for r in rows]
    return None

def _load_content_scores(path: str | None) -> dict[str, float]:
    """JSON: [{"property_id":"309","score":0.87}, ...] or {"309":0.87,...}"""
    if not path:
        return {}
    data = json.load(open(path, encoding="utf-8-sig"))  # tolerate BOM
    if isinstance(data, dict):
        return {str(k): float(v) for k, v in data.items()}
    if isinstance(data, list):
        out = {}
        for d in data:
            if isinstance(d, dict) and "property_id" in d and "score" in d:
                out[str(d["property_id"])] = float(d["score"])
        return out
    return {}

# === Args ===
parser = argparse.ArgumentParser()
parser.add_argument("--user-id", required=True, help="User ID (e.g. device_1)")
parser.add_argument("--k", type=int, default=10, help="Top-K recommendations")

# Candidate-ranking mode
parser.add_argument("--candidates", type=str, help="Comma-separated property_id list")
parser.add_argument("--candidates-file", type=str, help="Path to candidates file (csv/json)")

# Item cold-start fallback within candidates
parser.add_argument("--content-scores-file", type=str, default=None, help="JSON with content scores for items")
parser.add_argument("--fallback", type=str, choices=["mean", "zero", "constant"], default="mean",
                    help="Fallback for unseen items when ranking candidates (if no content scores given)")
parser.add_argument("--const", type=float, default=0.5, help="Constant score used if --fallback=constant")
args = parser.parse_args()

user_id = args.user_id
TOP_K = args.k
candidates = _load_candidates_from_args(args)
content_scores = _load_content_scores(args.content_scores_file)

# === Mode A: Cold-start user ===
if user_id not in user2idx:
    payload = {"user_id": user_id, "topk": TOP_K, "cold_start": True}
    if candidates:  # rank candidates by content score (or fallback)
        cand_scores = np.array([content_scores.get(pid, 0.0) for pid in candidates], dtype=float)
        norm = _minmax_norm(cand_scores)
        ranked = sorted(
            [{"property_id": pid, "score_content": float(s), "score_norm": float(n)}
             for pid, s, n in zip(candidates, cand_scores, norm)],
            key=lambda x: x["score_norm"], reverse=True
        )
        payload["recommendations"] = ranked[:TOP_K]
    else:
        # No candidates: use popularity fallback if available
        if item_popularity:
            top = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
            payload["recommendations"] = [{"property_id": pid, "pop_score": float(s)} for pid, s in top]
        else:
            # Last resort: arbitrary first K ids (not recommended but safe)
            payload["recommendations"] = [{"property_id": pid} for pid in list(item2idx.keys())[:TOP_K]]
    print(payload)
    raise SystemExit(0)

# === Mode B: Known user ===
u = user2idx[user_id]

# B1) No candidates -> global recommend (full catalog)
if not candidates:
    try:
        item_idx, scores = model.recommend(
            userid=u,
            user_items=None,
            N=TOP_K,
            filter_already_liked_items=False,
            recalculate_user=False
        )
    except Exception as e:
        print(f"⚠️ Recommendation failed: {e}")
        raise SystemExit(1)

    scores = np.array(scores, dtype=float)
    norm_scores = _minmax_norm(scores)
    recs = [
        {"property_id": idx2item[int(i)], "score_raw": float(s), "score_norm": float(sn)}
        for i, s, sn in zip(item_idx, scores, norm_scores)
    ]
    print({"user_id": user_id, "topk": TOP_K, "cold_start": False, "recommendations": recs})
    raise SystemExit(0)

# B2) Candidates provided -> rank within candidates, with unseen-item fallback
cand_known_idx, cand_known_ids, cand_unknown_ids = [], [], []
for pid in candidates:
    sp = str(pid)
    if sp in item2idx:
        cand_known_ids.append(sp)
        cand_known_idx.append(item2idx[sp])
    else:
        cand_unknown_ids.append(sp)

known_scores = []
known_ids_ordered = []

# 已见候选：用 recommend(..., items=selected) 代替 rank_items（避免弃用）
if cand_known_idx:
    try:
        selected = np.array(cand_known_idx, dtype=np.int32)
        item_idx, scores = model.recommend(
            userid=u,
            user_items=None,
            N=len(selected),
            items=selected,                    # 只对这些候选打分
            filter_already_liked_items=False,
            recalculate_user=False
        )
    except Exception as e:
        print(f"⚠️ recommend(items=...) failed: {e}")
        raise SystemExit(1)
    known_scores = np.array(scores, dtype=float)
    known_ids_ordered = [idx2item[int(i)] for i in item_idx]

# ========== 统一标尺归一化（关键改动） ==========
# 1) 先凑“整批候选”的原始分：已见用 CF raw，未见用内容 raw（若无内容则用 fallback base）
combined_ids = []
combined_raw = []

# 已见候选（CF raw）
for pid, raw in zip(known_ids_ordered, known_scores):
    combined_ids.append(pid)
    combined_raw.append(float(raw))

# 未见候选（内容 raw；若没有内容分，则用 fallback base）
fallback_scores = {}
if cand_unknown_ids:
    if content_scores:  # 优先内容分
        for pid in cand_unknown_ids:
            combined_ids.append(pid)
            combined_raw.append(float(content_scores.get(pid, 0.0)))
    else:
        # 没内容分 -> 用 mean/zero/constant
        # 用已见候选的 CF raw 求 mean；如果没有已见候选，就用 0.5
        if args.fallback == "mean":
            base = float(np.mean(known_scores)) if len(known_scores) else 0.5
        elif args.fallback == "zero":
            base = 0.0
        else:
            base = float(args.const)
        for pid in cand_unknown_ids:
            fallback_scores[pid] = base
            combined_ids.append(pid)
            combined_raw.append(float(base))

combined_raw = np.array(combined_raw, dtype=float)

# 2) 对整批候选做一次 min-max 归一化
combined_norm = _minmax_norm(combined_raw)

# 3) 组装最终结果（seen_in_training 标记来自 item2idx 是否存在）
results = []
for pid, raw, nrm in zip(combined_ids, combined_raw, combined_norm):
    results.append({
        "property_id": pid,
        "score_raw": float(raw),      # 已见=CF raw；未见=内容 raw 或 fallback
        "score_norm": float(nrm),     # 统一标尺后的 0~1 分
        "seen_in_training": (pid in item2idx)
    })

# 4) 排序并输出
results.sort(key=lambda x: x["score_norm"], reverse=True)
print({
    "user_id": user_id,
    "topk": TOP_K,
    "cold_start": False,
    "mode": "candidate_ranking",
    "recommendations": results[:TOP_K]
})
