# SystemCode/backend/app/dataservice/hybrid_utils.py
from typing import List, Dict
import numpy as np

def normalize_scores(rows: List[Dict]) -> List[Dict]:
    """
    输入: [{"property_id": "...", "score": 任意实数}, ...]
    输出: [{"property_id": "...", "score": 0~1}, ...]
    """
    if not rows:
        return rows
    arr = np.array([float(r["score"]) for r in rows], dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return [{"property_id": r["property_id"], "score": 0.5} for r in rows]
    out = (arr - lo) / (hi - lo + 1e-8)
    return [{"property_id": r["property_id"], "score": float(s)} for r, s in zip(rows, out)]
