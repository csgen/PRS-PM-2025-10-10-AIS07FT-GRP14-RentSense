# SystemCode/backend/app/dataservice/cf_service.py
from joblib import load
import json, os
import numpy as np
from typing import Iterable, List, Dict

MODEL_DIR = os.path.join("SystemCode", "backend", "app", "dataservice", "models")
ALS_PATH = os.path.join(MODEL_DIR, "als_model.pkl")
U2I_PATH = os.path.join(MODEL_DIR, "user2idx.json")
I2I_PATH = os.path.join(MODEL_DIR, "item2idx.json")

# --- 模型常驻内存：进程启动时只加载一次 ---
_model = load(ALS_PATH)
_user2idx = json.load(open(U2I_PATH, encoding="utf-8"))
_item2idx = json.load(open(I2I_PATH, encoding="utf-8"))
_idx2item = {v: k for k, v in _item2idx.items()}

def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.ones_like(arr, dtype=float) * 0.5
    return (arr - lo) / (hi - lo + 1e-8)

def recommend_cf_for_candidates(
    user_id: str,
    candidates: Iterable[str],
    topk: int | None = None,
    fallback: float = 0.5,  # 未出现在训练集里的房源的兜底分
    normalize: bool = True,
) -> List[Dict]:
    """
    同步函数：只对给定候选集打 CF 分。
    返回: [{"property_id": "...", "score": 0~1, "seen_in_training": bool}, ...]
    """
    cand_ids, cand_idx, unknown_ids = [], [], []
    for pid in candidates:
        sp = str(pid)
        if sp in _item2idx:
            cand_ids.append(sp)
            cand_idx.append(_item2idx[sp])
        else:
            unknown_ids.append(sp)

    # 新用户：CF 无法打分 -> 全部用兜底
    if user_id not in _user2idx:
        out = [{"property_id": pid, "score": float(fallback), "seen_in_training": False}
               for pid in (list(cand_ids) + list(unknown_ids))]
        return out[:topk] if topk else out

    u = _user2idx[user_id]

    known_scores = np.array([])
    known_ids_ordered: list[str] = []
    if cand_idx:
        # 只对“已见过的候选”打分（rank_items）
        item_idx, scores = _model.rank_items(
            userid=u,
            user_items=None,
            selected_items=np.array(cand_idx, dtype=np.int32),
            recalculate_user=False,
        )
        known_scores = np.array(scores, dtype=float)
        known_ids_ordered = [_idx2item[int(i)] for i in item_idx]

    # 归一化“已见候选”的分数
    norm = _minmax_norm(known_scores) if normalize else known_scores

    results: List[Dict] = []
    for pid, s in zip(known_ids_ordered, norm):
        results.append({"property_id": pid, "score": float(s), "seen_in_training": True})
    # 对“未见候选（训练集中未出现的房源）”给兜底分
    for pid in unknown_ids:
        results.append({"property_id": pid, "score": float(fallback), "seen_in_training": False})

    # 排序、截取
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:topk] if topk else results
