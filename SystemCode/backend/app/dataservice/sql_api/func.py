from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from sqlalchemy import func, text, or_
import time
import math
import numpy as np
import asyncio
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from .api_model import RequestInfo, ResultInfo
from .model import HousingData, District, University, CommuteTime, Park, HawkerCenter, Supermarket, Library, ImageRecord
from .envconfig import get_database_url_async, get_neo4j_info
from app.content_rcmd.embedding import scaler, encoder, global_means, autoencoder_user_req_emb

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

def get_total_score(price_norm, commute_norm, neighbourhood_norm, request):
    '''加权总分'''
    if request.importance_rent:
        price_score = request.importance_rent * price_norm
    if request.importance_location:
        commute_score = request.importance_location * commute_norm
    if request.importance_facility:
        neighbourhood_score = request.importance_facility * neighbourhood_norm

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
        # 若结果少于150条，补充至不少于150条
        target_count = 150
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

def cb_cf_rank(housings: list[HousingData], request: RequestInfo, w_content=0.7, w_cf=0.3, top_k=50):
    # TODO: 算相似度（用两组归一化分数）
    content_scores = user_housing_similarity_score(housings, request)
    cf_scores = user_housing_cf_score(housings, request)

    content_dict = {x["property_id"]: x["score"] for x in content_scores}
    cf_dict = {x["property_id"]: x["score"] for x in cf_scores}

    merged = []
    for pid, c_score in content_dict.items():
        cf_score = cf_dict.get(pid, 0)  # 没有视为0
        final_score = w_content * c_score + w_cf * cf_score
        merged.append({
            "property_id": pid,
            "score": final_score,
            "source": "content+cf"
        })

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]
    

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
            total_score = get_total_score(price_norm[i],commute_norm[i],neighbour_norm[i],request)

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
            results.append((resultInfo, total_score))

    # 排序取前 50
    results_sorted = sorted(results, key=lambda pair: pair[1], reverse=True)[:50]
    return [pair[0] for pair in results_sorted]

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
    price_val = (request.min_monthly_rent + request.max_monthly_rent) / 2
    price_scaled = scaler["price"].transform([[price_val]])[0][0]

    # 没有面积输入 → 用平均值
    area_scaled = scaler["area_sqft"].transform([[global_means["area_sqft"]]])[0][0]

    # 没有build_time输入 → 平均年份
    build_time_scaled = scaler["build_time"].transform([[global_means["build_time"]]])[0][0]

    # 距MRT距离
    if request.max_mrt_distance:
        mrt_scaled = scaler["distance_to_mrt"].transform([[request.max_mrt_distance]])[0][0]
    else:
        mrt_scaled = scaler["distance_to_mrt"].transform([[global_means["distance_to_mrt"]]])[0][0]
    # 先用单间
    beds_scaled = scaler["beds_num"].transform([[1]])[0][0]
    baths_scaled = scaler["baths_num"].transform([[1]])[0][0]

    num_vec = [price_scaled, area_scaled, build_time_scaled, mrt_scaled, beds_scaled, baths_scaled]

    # 类别层
    # 类型
    if request.flat_type_preference:
        type_vec = encoder["type"].transform(request.flat_type_preference).mean(axis=0)
    else:
        type_vec = np.zeros(len(encoder["type"].categories_[0]))
    # 区域
    if request.target_district_id:
        district_vec = encoder["district_id"].transform([[request.target_district_id]]).flatten()
    else:
        district_vec = np.zeros(len(encoder["district_id"].categories_[0]))
    # 其他
    is_room = 0
    build_missing = 0

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
        num_vec, type_vec, district_vec, [is_room, build_missing],
        text_vec, geo_vec
    ])

    return user_vec

# 用户输入embedding
def get_user_req_emb(request: RequestInfo):
    user_vec = get_user_req_feature_vector(request=request)
    emb = autoencoder_user_req_emb(user_vec=user_vec)
    return emb

def user_housing_similarity_score(housings: list[HousingData], request: RequestInfo):
    '''
    根据用户输入计算每个房源与用户 embedding 的相似度并打分
    输出：
    [ {"property_id": 87, "score": 0.913, "source": "content"},
    {"property_id": 102, "score": 0.908, "source": "content"},
    ... ]
    '''
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
    # scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

def user_housing_cf_score(housings: list[HousingData], request: RequestInfo):
    '''
    根据cf打分
    输出：
    [ {"property_id": 87, "score": 0.913, "source": "cf"},
    {"property_id": 102, "score": 0.908, "source": "cf"},
    ... ]
    '''
    # TODO: 真实cf逻辑 @ruyanjie
    scored = [
        {"property_id": int(housings[i].id), "score": 0, "source": "cf"}
        for i in range(len(housings))
    ]
    return scored