from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from sqlalchemy import func, text, or_
import time
import math
from .api_model import RequestInfo, ResultInfo
from .model import HousingData, District, University, CommuteTime, Park, HawkerCenter, Supermarket, Library, ImageRecord
from .envconfig import get_database_url_async

DATABASE_URL_ASYNC = get_database_url_async()
async_engine = create_async_engine(
    DATABASE_URL_ASYNC, 
    echo=False,)

AsyncSessionLocal = sessionmaker(
    bind=async_engine, class_=AsyncSession, expire_on_commit=False
)

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
        # 若结果少于50条，补充至50条
        target_count = 50
        if original_count < target_count:
            housings = await _expand_query_conditions(
                session=session,
                request=request,
                existing_ids=[h.id for h in housings],
                current_results=housings,
                target_count=target_count
            )
        print(f'第一步输出前一共{len(housings)}条房源')
        return housings[:target_count]

async def filter_housing_async(housings: list[HousingData], request: RequestInfo):
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