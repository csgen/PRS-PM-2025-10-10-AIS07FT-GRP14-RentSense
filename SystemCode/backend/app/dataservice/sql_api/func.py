from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from sqlalchemy import func, text, or_
import os
import json
import time
import math
import numpy as np
import pandas as pd
import asyncio
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from .api_model import RequestInfo, ResultInfo
from .model import HousingData, District, University, CommuteTime, Park, HawkerCenter, Supermarket, Library, ImageRecord
from .envconfig import get_database_url_async, get_neo4j_info
from app.content_rcmd.embedding import scaler, encoder, global_means, autoencoder_user_req_emb

from joblib import load


DATABASE_URL_ASYNC = get_database_url_async()
async_engine = create_async_engine(
    DATABASE_URL_ASYNC, 
    echo=False,)

AsyncSessionLocal = sessionmaker(
    bind=async_engine, class_=AsyncSession, expire_on_commit=False
)

# neo4j driver
url, username, password = get_neo4j_info()
graphdriver = GraphDatabase.driver(url, auth=(username, password))

# cf
# === CF (ALS) artifacts, lazy-loaded once ===
_CF_MODEL_DIR = os.path.join("app", "dataservice", "models")
_CF_ALS_PATH  = os.path.join(_CF_MODEL_DIR, "als_model.pkl")
_CF_U2I_PATH  = os.path.join(_CF_MODEL_DIR, "user2idx.json")
_CF_I2I_PATH  = os.path.join(_CF_MODEL_DIR, "item2idx.json")

_CF_MODEL = None
_CF_USER2IDX = None
_CF_ITEM2IDX = None
_CF_IDX2ITEM = None

def _ensure_cf_loaded():
    """Lazy load ALS model and mappings once."""
    global _CF_MODEL, _CF_USER2IDX, _CF_ITEM2IDX, _CF_IDX2ITEM
    if _CF_MODEL is not None:
        return
    if not (os.path.exists(_CF_ALS_PATH) and os.path.exists(_CF_U2I_PATH) and os.path.exists(_CF_I2I_PATH)):
        # 没有模型就保持 None；CF 会自动给出兜底分数（全 0）由内容分顶上
        print("⚠️ CF model artifacts not found, CF scores default to 0.")
        return
    print(f"[CF] Loading ALS model/mappings from: '{_CF_MODEL_DIR}'")
    _CF_MODEL = load(_CF_ALS_PATH)
    # 兼容 BOM
    _CF_USER2IDX = json.load(open(_CF_U2I_PATH, encoding="utf-8-sig"))
    _CF_ITEM2IDX = json.load(open(_CF_I2I_PATH, encoding="utf-8-sig"))
    _CF_IDX2ITEM = {v: k for k, v in _CF_ITEM2IDX.items()}

    # ---- CONSISTENCY SHRINK FOR ITEMS ----
    try:
        n_items = int(getattr(_CF_MODEL.item_factors, "shape", [0])[0])
    except Exception:
        n_items = 0
    
    if n_items > 0:
        # 只保留索引在 [0, n_items-1] 的条目
        _CF_ITEM2IDX = {pid: idx for pid, idx in _CF_ITEM2IDX.items() if 0 <= int(idx) < n_items}
        _CF_IDX2ITEM = {int(idx): pid for pid, idx in _CF_ITEM2IDX.items()}
        print(f"[CF] normalized mappings: items_in_model={n_items}, items_kept={len(_CF_ITEM2IDX)}")
    else:
        print("[CF] WARN: model has no item_factors? n_items=0")
        

def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.ones_like(arr, dtype=float) * 0.5
    return (arr - lo) / (hi - lo + 1e-12)
# === CF (ALS) artifacts, lazy-loaded once ===

async def get_district_centroids():
    async with AsyncSessionLocal() as session:
        stmt = select(District.id, District.latitude, District.longitude)
        result = await session.execute(stmt)
        rows = result.all()
        centroids = {r.id: (r.latitude, r.longitude) for r in rows if r.latitude and r.longitude}
        return centroids

_district_centroids_cache = None
# 区域位置缓存（同步/异步包装）
def get_district_centroids_cached_sync():
    global _district_centroids_cache
    if _district_centroids_cache is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # 如果当前已经有事件循环
            return loop.create_task(get_district_centroids())
        else:
            _district_centroids_cache = asyncio.run(get_district_centroids())
    return _district_centroids_cache
    
district_centroids = get_district_centroids_cached_sync()

def remove_duplicate_housings(housings: list[HousingData]) -> tuple[list[HousingData], int]:
    '''去除少量重复的房源记录，返回去重后的列表和去除的数量'''
    seen = set()
    unique_housings = []
    removed_count = 0
    
    for housing in housings:
        key = (
            housing.name,
            housing.price,
            housing.area_sqft,
            housing.type,
            housing.location,
            housing.distance_to_mrt,
            housing.beds_num,
            housing.baths_num
        )
        
        if key not in seen:
            seen.add(key)
            unique_housings.append(housing)
        else:
            removed_count += 1
    
    return unique_housings, removed_count

def get_importance_total_score(price_norm, commute_norm, neighbourhood_norm, request):
    '''加权总分'''
    i_rent = request.importance_rent or 3
    i_location = request.importance_location or 3
    i_facility = request.importance_facility or 3
    price_score = i_rent * price_norm
    commute_score = i_location * commute_norm
    neighbourhood_score = i_facility * neighbourhood_norm

    return price_score+commute_score+neighbourhood_score

async def query_housing_data_async(request: RequestInfo) -> list[HousingData]:
    '''根据 RequestInfo 查询符合条件的房源'''

    start_time = time.time()

    # 基础查询
    stmt = (
        select(HousingData)
        .join(CommuteTime, HousingData.id == CommuteTime.housing_id)
        .where(
            CommuteTime.university_id == request.school_id,
            HousingData.price >= request.min_monthly_rent,
            HousingData.price <= request.max_monthly_rent,
        )
    )

    # 动态条件
    if request.target_district_id is not None:
        stmt = stmt.where(HousingData.district_id == request.target_district_id)

    if request.max_school_limit is not None:
        stmt = stmt.where(CommuteTime.commute_time_minutes <= request.max_school_limit)

    if request.flat_type_preference:
        stmt = stmt.where(HousingData.type.in_(request.flat_type_preference))

    if request.max_mrt_distance is not None:
        stmt = stmt.where(HousingData.distance_to_mrt <= request.max_mrt_distance)

    # 异步执行查询
    async with AsyncSessionLocal() as session:
        result = await session.execute(stmt)
        housings = result.scalars().all()

        original_count = len(housings)
        print(f"在{time.time() - start_time:.2f} 秒内通过初步过滤得到{original_count}条房源记录。")
        # 若结果少于100条，补充至不少于100条
        target_count = 100
        if original_count < target_count:
            housings = await _expand_query_conditions(
                session=session,
                request=request,
                existing_ids=[h.id for h in housings],
                current_results=housings,
                target_count=target_count
            )
        print(f'第一步输出一共{len(housings)}条房源')
        return housings

def merge_importance_cb_cf_score(importance_norms, housings: list[HousingData], request: RequestInfo, w_importance=0.4, w_content=0.4, w_cf=0.2, top_k=50):
    content_scores = user_housing_similarity_score(housings, request)
    content_dict = {x["property_id"]: x["score"] for x in content_scores}

    cf_scores = user_housing_cf_score(housings, request)
    cf_dict = {x["property_id"]: x["score"] for x in cf_scores}

    all_content = [content_dict.get(h.id, 0) for h in housings]
    all_cf = [cf_dict.get(h.id, 0) for h in housings]

    merged = []
    for i in range(len(housings)):
        final_score = w_importance * importance_norms[i] + w_content * all_content[i] + w_cf * all_cf[i]
        merged.append({
            'id': housings[i].id,
            'final_score': final_score,
            'source': 'importance+content+cf'
        })

    return merged

async def filter_housing_async(housings: list[HousingData], request: RequestInfo)->list[ResultInfo]:
    '''根据 RequestInfo 对所有房源进行过滤并计算评分'''
    results = []
    
    async with AsyncSessionLocal() as session:
        raw_data = []
        radius_m = 2000

        housing_ids = [h.id for h in housings]
        district_ids = list({h.district_id for h in housings if h.district_id})
        
        # 批量取 District
        district_map = {
            d.id: d for d in (
                await session.execute(select(District).where(District.id.in_(district_ids)))
            ).scalars().all()
        }
        
        # 批量取 Image
        image_map = {
            i.id: i.public_url for i in (
                await session.execute(select(ImageRecord.id, ImageRecord.public_url)
                                    .where(ImageRecord.id.in_(housing_ids)))
            ).all()
        }
        
        # 批量取 Commute
        commute_map = {
            c.housing_id: c.commute_time_minutes for c in (
                await session.execute(
                    select(CommuteTime.housing_id, CommuteTime.commute_time_minutes)
                    .where(CommuteTime.housing_id.in_(housing_ids),
                        CommuteTime.university_id == request.school_id)
                )
            ).all()
        }

        async def get_facilities_from_cache(session: AsyncSession, housing_ids: list[int], radius_m: int = 2000):
            '''
            从房源设施距离表 housing_facility_distances 获取每个房源 2km 内最近的设施（按类型分组）
            返回格式：
            {
                housing_id: [
                    {"some market": "1234"},
                    {"some park": "987"},
                    ...
                ]
            }
            '''
            stmt = text("""
                SELECT DISTINCT ON (housing_id, facility_type)
                    housing_id,
                    facility_type,
                    facility_name,
                    distance_m
                FROM housing_facility_distances
                WHERE housing_id = ANY(:housing_ids)
                AND distance_m <= :radius_m
                ORDER BY housing_id, facility_type, distance_m;
            """)

            rows = await session.execute(stmt, {"housing_ids": housing_ids, "radius_m": radius_m})
            rows = rows.mappings().all()

            facility_map: dict[int, list[dict]] = {}

            for row in rows:
                hid = row["housing_id"]
                if hid not in facility_map:
                    facility_map[hid] = []
                facility_map[hid].append({
                    row["facility_name"]: str(int(row["distance_m"])),
                })

            return facility_map

        start_time = time.time()
        
        facility_map = await get_facilities_from_cache(session, housing_ids, radius_m=2000)
        
        print(f'设施查询时间: {time.time() - start_time:.2f} 秒')

        # 处理每个房源
        def process_housing(housing: HousingData):
            district = district_map.get(housing.district_id)
            district_safety_score = district.safety_score if district else 0.0
            district_name = district.district_name if district else ""

            img_url = image_map.get(housing.id)
            commute_time = commute_map.get(housing.id)

            # 从预查询的结果中获取设施信息
            nearest_facilities = facility_map.get(housing.id, [])
            
            facility_score = len(nearest_facilities)

            return {
                "housing": housing,
                "img": img_url,
                "price": housing.price or 0,
                "commute": commute_time or 9999,
                "facility": facility_score,
                "safety": district_safety_score or 0,
                "district": district_name,
                "public_facilities": nearest_facilities
            }

        raw_data = [process_housing(h) for h in housings]
        
        execution_time = time.time() - start_time
        print(f'总查询时间: {execution_time:.2f} 秒')

        # === 归一化函数 ===
        def normalize(values, reverse=False):
            vals = [v for v in values if v is not None]
            if not vals:
                return [0 for _ in values]
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                return [1 for _ in values]
            return [
                (max_v - v) / (max_v - min_v) if reverse else (v - min_v) / (max_v - min_v)
                for v in values
            ]

        price_norm = normalize([x["price"] for x in raw_data], reverse=True) # 价格越低越好
        commute_norm = normalize([x["commute"] for x in raw_data], reverse=True) # 通勤时间越短越好
        facility_norm = normalize([x["facility"] for x in raw_data])
        safety_norm = normalize([x["safety"] for x in raw_data])
        neighbour_norm = normalize([(facility_norm[i] * 2 + safety_norm[i]) for i in range(len(raw_data))]) # 邻里综合评分

        for i, data in enumerate(raw_data):
            housing = data["housing"]
            importance_score = get_importance_total_score(price_norm[i],commute_norm[i],neighbour_norm[i],request)

            resultInfo = ResultInfo(
                property_id=housing.id,
                img_src=data['img'],
                name=housing.name,
                district=data['district'],
                price=str(housing.price),
                beds=housing.beds_num,
                baths=housing.baths_num,
                area=housing.area_sqft,
                build_time=str(housing.build_time) if housing.build_time else "",
                location=housing.location,
                time_to_school=int(data["commute"]),
                distance_to_mrt=int(housing.distance_to_mrt) if housing.distance_to_mrt else None,
                latitude=housing.latitude,
                longitude=housing.longitude,
                public_facilities=data["public_facilities"],
                facility_type=housing.type,
                costScore=round(price_norm[i], 2),
                commuteScore=round(commute_norm[i], 2),
                neighborhoodScore=round(neighbour_norm[i], 2)
            )
            results.append((resultInfo, importance_score))

    # 总分归一化
    importance_total_norms = normalize([x[1] for x in results])

    merged_score = merge_importance_cb_cf_score(importance_total_norms, housings, request)

    # 将 merged_score 的 final_score 映射为 {id: score}
    merged_dict = {m["id"]: m["final_score"] for m in merged_score}

    results_sorted = sorted(
        results,
        key=lambda pair: merged_dict.get(pair[0].property_id, 0.0),
        reverse=True
    )

    # 排序取前 50
    top_results = [pair[0] for pair in results_sorted[:50]]
    return top_results

# 辅助函数：查询邻近的 district_id
async def _find_nearby_districts(session: AsyncSession, district_id) -> list[int]:
    '''根据地理位置找到距离目标 district 最近的 1-2 个相邻区域'''
    
    start_time = time.time()
    if not district_id:
        return []

    # 获取目标 district 的地理位置
    center_query = (
        select(District.id, District.geog)
        .where(District.id == district_id)
    )
    center_row = (await session.execute(center_query)).first()
    if not center_row or not center_row.geog:
        print(f"District {district_id} 没有地理信息，无法计算邻近区域。")
        return []

    center_geog = center_row.geog

    # 查询距离最近的 2 个 district
    # 指定半径内，再按 ST_Distance 排序
    nearby_stmt = (
        select(District.id)
        .where(District.id != district_id)
        .where(func.ST_DWithin(District.geog, center_geog, 10000))  # 10000米内
        .order_by(func.ST_Distance(District.geog, center_geog))
        .limit(2)
    )

    nearby_ids = (await session.execute(nearby_stmt)).scalars().all()

    print(f"District {district_id} 邻近区域ID: {nearby_ids}")
    print(f'放宽区域查询时间: {time.time() - start_time:.2f} 秒')
    return nearby_ids

# 辅助函数：根据 importance 动态放宽租金与通勤时间
async def _expand_by_importance(
    session: AsyncSession,
    request: RequestInfo,
    results: list[HousingData],
    existing_ids: list[int],
    target_count: int
) -> list[HousingData]:
    '''
    importance_rent 越低，放宽幅度越大；
    importance_location 越低，放宽幅度越大
    '''
    start_time = time.time()
    rent_loosen_factor = (6 - (request.importance_rent or 3)) / 5  # 范围 0~1
    location_loosen_factor = (6 - (request.importance_location or 3)) / 5

    # 放宽比例
    rent_expand_pct = 0.2 + 0.4 * rent_loosen_factor  # 租金放宽比例 20%~60%
    commute_expand_pct = 0.2 + 0.4 * location_loosen_factor  # 通勤放宽比例 20%~60%

    min_rent = math.floor(request.min_monthly_rent * (1 - rent_expand_pct) * 0.5) # 租金下限直接放宽两倍
    max_rent = math.ceil(request.max_monthly_rent * (1 + rent_expand_pct))

    commute_limit = (
        request.max_school_limit * (1 + commute_expand_pct)
        if request.max_school_limit else None
    )

    stmt = (
        select(HousingData)
        .join(CommuteTime, HousingData.id == CommuteTime.housing_id)
        .where(
            CommuteTime.university_id == request.school_id,
            HousingData.price >= min_rent,
            HousingData.price <= max_rent,
            HousingData.id.notin_(existing_ids),
        )
    )
    if commute_limit:
        stmt = stmt.where(CommuteTime.commute_time_minutes <= commute_limit)

    stmt = stmt.order_by(CommuteTime.commute_time_minutes.asc())
    res = (await session.execute(stmt)).scalars().all()
    print(f"根据 importance 放宽条件，新增 {len(res)} 条")
    results.extend(res)
    print(f'放宽租金与通勤限制查询时间: {time.time() - start_time:.2f} 秒')
    return results

# 辅助函数：放宽条件逐步补充（过滤数量太少的情况下）
async def _expand_query_conditions(
    session: AsyncSession,
    request: RequestInfo,
    existing_ids: list[int],
    current_results: list[HousingData],
    target_count: int
) -> list[HousingData]:
    """
    按优先级逐步放宽查询条件，直到结果达到 target_count。
    """
    results = list(current_results)
    print("进入放宽策略...")

    start_time = time.time()
    # 1. 放宽区域限制（邻近district），并不再限制type
    nearby_districts = await _find_nearby_districts(session, request.target_district_id)
    stmt = (
        select(HousingData)
        .join(CommuteTime, HousingData.id == CommuteTime.housing_id)
        .where(
            CommuteTime.university_id == request.school_id,
            HousingData.id.notin_(existing_ids),
            or_(
                HousingData.district_id.in_(nearby_districts + [request.target_district_id])
                if nearby_districts else True,
            ),
        )
    )

    res = (await session.execute(stmt)).scalars().all()
    print(f"放宽地理范围 + flat_type 限制，新增 {len(res)} 条")
    results.extend(res)
    existing_ids.extend([r.id for r in res])

    # 若仍不足目标数量，继续放宽价格与通勤限制
    if len(results) < target_count:
        results = await _expand_by_importance(session, request, results, existing_ids, target_count)

    # 最后还不足：按通勤时间最短补齐
    if len(results) < target_count:
        fallback_stmt = (
            select(HousingData)
            .join(CommuteTime, HousingData.id == CommuteTime.housing_id)
            .where(
                CommuteTime.university_id == request.school_id,
                HousingData.id.notin_(existing_ids),
            )
            .order_by(CommuteTime.commute_time_minutes.asc())
            .limit(target_count - len(results))
        )
        fallback = (await session.execute(fallback_stmt)).scalars().all()
        print(f"兜底补充 {len(fallback)} 条房源。")
        results.extend(fallback)

    print(f'放宽限制查询时间: {time.time() - start_time:.2f} 秒')
    return results

# 用户输入向量化
def get_user_req_feature_vector(request: RequestInfo):
    # 数值层
    # 租金区间 → 理想租金
    # price -> 租金区间的中点
    price_val = (request.min_monthly_rent + request.max_monthly_rent) / 2

    # area_sqft -> 用户没填，假设用平均面积
    area_val = global_means["area_sqft"]

    # build_time -> 用户没提供；用平均建成年份
    build_time_val = global_means["build_time"]

    # distance_to_mrt -> 如果用户提供 max_mrt_distance 就用，否则用平均
    if request.max_mrt_distance is not None:
        mrt_val = request.max_mrt_distance
    else:
        mrt_val = global_means["distance_to_mrt"]

    # beds_num / baths_num -> 默认 1+0(单间)
    beds_val = 1
    baths_val = 0

    # 构造一行 dataframe
    num_df = pd.DataFrame([[
        price_val,
        area_val,
        build_time_val,
        mrt_val,
        beds_val,
        baths_val
    ]], columns=["price", "area_sqft", "build_time", "distance_to_mrt", "beds_num", "baths_num"])

    num_scaled = scaler.transform(num_df)  # shape: (1, 6)
    # num_vec shape -> (6,)
    num_vec = num_scaled[0]

    # 类别层
    # 类型
    if request.flat_type_preference and len(request.flat_type_preference) > 0:
        # 对每个偏好类别进行 one-hot，然后取平均
        pref_types = np.vstack([
            encoder.transform([[ft, None, 0, 0]])[0][:len(encoder.categories_[0])]
            for ft in request.flat_type_preference
        ])
        type_vec = pref_types.mean(axis=0)
    else:
        # 如果没有偏好，则取类别均值（即每个类别权重相等）
        n_type_cats = len(encoder.categories_[0])
        type_vec = np.ones(n_type_cats) / n_type_cats
    # 区域
    district_val = request.target_district_id if request.target_district_id is not None else None
    # 其他
    is_room_val = 0
    build_time_missing_val = 0

    # 只对3个特征进行编码（不含type）
    # 手动拼接
    partial_encoded = encoder.transform([[None, district_val, is_room_val, build_time_missing_val]])[0]
    # 截掉前面 type 部分的维度
    offset = len(encoder.categories_[0])
    cat_rest_vec = partial_encoded[offset:]  # district_id + is_room + build_time_missing

    # 拼接最终类别层
    cat_vec = np.concatenate([type_vec, cat_rest_vec])

    # 文本层
    text_vec = np.zeros(384)

    # 空间层
    if request.target_district_id and request.target_district_id in district_centroids:
        lat, lon = district_centroids[request.target_district_id]
    else:
        lat, lon = global_means["latitude"], global_means["longitude"]
    
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    geo_vec = np.array([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r)
    ])

    user_vec = np.concatenate([
        num_vec,      # (6,)
        cat_vec,      # (43,)
        text_vec,     # (384,)
        geo_vec       # (3,)
    ]).astype(np.float32)

    return user_vec

# 用户输入embedding
def get_user_req_emb(request: RequestInfo):
    start_time = time.time()

    user_vec = get_user_req_feature_vector(request=request)
    emb = autoencoder_user_req_emb(user_vec=user_vec)
    
    execution_time = time.time() - start_time
    print(f'用户输入向量embedding执行时间: {execution_time:.2f} 秒')

    return emb

def user_housing_similarity_score(housings: list[HousingData], request: RequestInfo):
    '''
    根据用户输入计算每个房源与用户 embedding 的相似度并打分
    输出：
    [ {"property_id": 87, "score": 0.913, "source": "content"},
    {"property_id": 102, "score": 0.908, "source": "content"},
    ... ]
    '''
    start_time = time.time()

    user_emb = get_user_req_emb(request=request)
    user_emb = np.array(user_emb).reshape(1, -1)
    housing_ids = [h.id for h in housings]
    embeddings = []
    with graphdriver.session() as session:
        result = session.run(
            """
            MATCH (p:Property)
            WHERE p.id IN $ids
            RETURN p.id AS id, p.embedding AS embedding
            """,
            ids=housing_ids
        )
        records = result.data()
    graphdriver.close()
    emb_dict = {r["id"]: np.array(r["embedding"], dtype=np.float32) for r in records}
    embeddings = [emb_dict.get(h.id, np.zeros(128)) for h in housings]
    property_embs = np.stack(embeddings)

    # 相似度计算
    similarities = cosine_similarity(user_emb, property_embs)[0]

    scored = [
        {"property_id": int(housings[i].id), "score": float(similarities[i]), "source": "content"}
        for i in range(len(housings))
    ]

    execution_time = time.time() - start_time
    print(f'计算cosine_similarity执行时间: {execution_time:.2f} 秒')
    return scored




def user_housing_cf_score(housings: list[HousingData], request: RequestInfo):
    """
    用已训练 ALS 模型对候选 housings 打 CF 分。
    - 已见房源（在 item2idx 中出现过）用 ALS 的 recommend(items=...) 打 raw 分
    - 未见房源（训练集中没出现过的 ID）给 0（或可改成 mean/constant，交给内容模型兜底）
    - 新用户（user_id 不在 user2idx）整批返回 0（让内容分接管）
    返回：
      [{"property_id": <int>, "score": 0~1, "source": "cf"}, ...]
    """
    import traceback, numpy as np

    try:
        # --- Step 1: 确保模型已加载 ---
        _ensure_cf_loaded()
        print(f"[CF] ✅ model loaded ok | users={len(_CF_USER2IDX)} items={len(_CF_ITEM2IDX)}")

        # --- Step 2: 获取 user_id ---
        user_id = getattr(request, "device_id", None) or getattr(request, "user_id", None)
        print(f"[CF] user_id={user_id}")
        if not user_id or (user_id not in _CF_USER2IDX):
            print("[CF] ⚠️ cold-start user -> CF score = 0")
            return [{"property_id": int(h.id), "score": 0.0, "source": "cf"} for h in housings]
        u = _CF_USER2IDX[user_id]

        # --- Step 3: 分出已见/未见房源 ---
        cand_ids_str = [str(h.id) for h in housings]
        known_item_idx, known_item_pids, unknown_pids = [], [], []
        for pid in cand_ids_str:
            if pid in _CF_ITEM2IDX:
                known_item_pids.append(pid)
                known_item_idx.append(_CF_ITEM2IDX[pid])
            else:
                unknown_pids.append(pid)

        print(f"[CF] candidates: {len(housings)} | known={len(known_item_idx)} | unknown={len(unknown_pids)}")

        # --- Step 4: 已见候选打分 ---
        pid_to_cfraw = {}
        if known_item_idx:
            selected = np.array(known_item_idx, dtype=np.int32)
            try:
                item_idx, scores = _CF_MODEL.recommend(
                    userid=u,
                    user_items=None,
                    N=len(selected),
                    items=selected,
                    filter_already_liked_items=False,
                    recalculate_user=False
                )
                scores = np.array(scores, dtype=float)
                ordered_pids = [_CF_IDX2ITEM[int(i)] for i in item_idx]
                for pid, raw in zip(ordered_pids, scores):
                    pid_to_cfraw[pid] = float(raw)
                print(f"[CF] scored known items OK, sample={list(pid_to_cfraw.items())[:5]}")
            except Exception as e:
                print(f"[CF] ❌ recommend() failed: {repr(e)}")
                traceback.print_exc()
                return [{"property_id": int(h.id), "score": 0.0, "source": "cf"} for h in housings]
        else:
            print("[CF] ⚠️ no known items, all 0")

        # --- Step 5: 未见候选填 0 ---
        for pid in unknown_pids:
            pid_to_cfraw[pid] = 0.0

        # --- Step 6: 归一化 ---
        raw_list = np.array([pid_to_cfraw.get(str(h.id), 0.0) for h in housings], dtype=float)
        if raw_list.max() == raw_list.min():
            norm_list = np.zeros_like(raw_list)
        else:
            norm_list = (raw_list - raw_list.min()) / (raw_list.max() - raw_list.min())

        # --- Step 7: 返回结果 ---
        scored = [
            {"property_id": int(h.id), "score": float(n), "source": "cf"}
            for h, n in zip(housings, norm_list)
        ]
        print(f"[CF] ✅ finished scoring | sample={scored[:3]}")
        return scored

    except Exception as e:
        print("[CF] ❌ Exception in user_housing_cf_score:", repr(e))
        traceback.print_exc()
        return [{"property_id": int(h.id), "score": 0.0, "source": "cf"} for h in housings]
