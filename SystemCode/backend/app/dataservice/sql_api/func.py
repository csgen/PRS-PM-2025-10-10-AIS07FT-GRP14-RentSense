from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from sqlalchemy import func, text
import time
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
        print(DATABASE_URL_ASYNC)
        result = await session.execute(stmt)
        housings = result.scalars().all()

        original_count = len(housings)
        print(f"初步过滤得到{original_count}条房源记录。")

        # 若结果少于50条，补充至通勤时间最短的50条（不重复）
        target_count = 50
        if original_count < target_count:
            existing_ids = [h.id for h in housings]

            fallback_stmt = (
                select(HousingData)
                .join(CommuteTime, HousingData.id == CommuteTime.housing_id)
                .where(
                    CommuteTime.university_id == request.school_id,
                    HousingData.id.notin_(existing_ids),
                )
                .order_by(CommuteTime.commute_time_minutes.asc())
                .limit(target_count-original_count)
            )

            fallback_result = await session.execute(fallback_stmt)
            fallback_housings = fallback_result.scalars().all()
            print(f"补充了{len(fallback_housings)}条房源记录以满足最小数量要求。")
            housings.extend(fallback_housings)

        # 少量去重并返回
        housings, removed_count = remove_duplicate_housings(housings)
        return_count = min(len(housings), target_count)
        return housings[:return_count]

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

