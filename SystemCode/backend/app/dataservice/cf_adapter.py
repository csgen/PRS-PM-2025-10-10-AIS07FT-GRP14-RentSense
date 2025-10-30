# SystemCode/backend/app/dataservice/cf_adapter.py
import asyncio
from typing import List, Dict, Iterable
from SystemCode.backend.app.dataservice.cf_service import recommend_cf_for_candidates

async def collaborative_filter_async(
    device_id: str,
    candidates: Iterable[str],
    topk: int | None = None,
) -> List[Dict]:
    """
    异步封装：线程池里调用同步 CF 函数，避免阻塞事件循环。
    返回: [{"property_id": "...", "score": 0~1}, ...]
    """
    loop = asyncio.get_event_loop()
    recs = await loop.run_in_executor(
        None,
        lambda: recommend_cf_for_candidates(
            user_id=device_id,
            candidates=candidates,
            topk=topk,
            fallback=0.5,     # 训练未见过的房源给 0.5 当兜底
            normalize=True,   # 输出 0~1，方便融合
        )
    )
    # 统一字段：只保留融合需要的键
    return [{"property_id": r["property_id"], "score": float(r["score"])} for r in recs]
