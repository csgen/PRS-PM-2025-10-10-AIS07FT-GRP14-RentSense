import requests
import time
from sqlalchemy import create_engine, or_, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import asyncio
from geoalchemy2.shape import from_shape
from shapely.geometry import Point
import re
import time
from envconfig import get_database_url, get_openmap_token, get_openmap_library_url, get_database_url_async
from model import Base, District, HousingData, University, CommuteTime, Library, Park, HawkerCenter, Supermarket

database_url = get_database_url()
engine = create_engine(database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def update_all_npc_locations():
    '''更新所有警署的地理信息'''
    session = SessionLocal()
    try:
        districts = session.query(District).all()

        print(f"Found {len(districts)} districts to update.")

        for d in districts:
            lon, lat = get_longitude_latitude(d.postal_code)
            if lon and lat:
                d.longitude = lon
                d.latitude = lat

                d.geog = from_shape(Point(lon, lat), srid=4326)
                d.geom = from_shape(Point(lon, lat), srid=3414)

                print(f"✅ Updated {d.neighbour_police_center} ({lat}, {lon})")

            time.sleep(0.3)

        session.commit()
        print("All locations updated successfully.")
    except Exception as e:
        session.rollback()
        print("Error:", e)
    finally:
        session.close()

def get_longitude_latitude(query: str):
    '''使用OneMap API根据查询获取经纬度'''
    url_1, url_2, _, token = get_openmap_token()
    url = f'{url_1}{query}{url_2}'
    headers = {"Authorization":token}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if data["found"] > 0:
            result = data["results"][0]
            # print(result)
            lon = float(result["LONGITUDE"])
            lat = float(result["LATITUDE"])
            return lon, lat
    except Exception as e:
        print(f"Error fetching postal code {query}: {e}")
    return None, None

def update_all_housing_locations():
    '''更新所有房源的地理信息'''
    session = SessionLocal()
    try:
        housings = session.query(HousingData).all()

        print(f"Found {len(housings)} housings to update.")

        failcount = 0
        for h in housings:
            location = h.location
            # 替换中文单引号为英文单引号
            location = location.replace('‘', "'").replace('’', "'")
            # 去掉开头的 "near "
            location = re.sub(r'^near\s+', '', location, flags=re.IGNORECASE)
            # 去掉一些符号及其后的内容
            pattern = r'[\(,/,·].*'
            location = re.sub(pattern, '', location)
            # 去除开头和结尾的空格
            location = location.strip()
            lon, lat = get_longitude_latitude(location)
            if lon and lat:
                h.longitude = lon
                h.latitude = lat

                h.geog = from_shape(Point(lon, lat), srid=4326)
                h.geom = from_shape(Point(lon, lat), srid=3414)

                # print(f"✅ Updated {d.neighbour_police_center} ({lat}, {lon})")
            else:
                print(f'❌ Failed to get location for: {h.id} : {location}')
                failcount += 1

            time.sleep(0.3)

        session.commit()
        print("All locations updated successfully.")
        print(f"Failed to update {failcount} records.")
    except Exception as e:
        session.rollback()
        print("Error:", e)
    finally:
        session.close()

def find_housing_district():
    query = text("""
    UPDATE housing_data AS h
    SET district_id = d.id
    FROM (
        SELECT h2.id AS hid, d2.id AS id
        FROM housing_data h2
        JOIN LATERAL (
            SELECT id
            FROM districts
            ORDER BY h2.geog <-> districts.geog
            LIMIT 1
        ) d2 ON true
    ) AS d
    WHERE h.id = d.hid;
    """)
    start = time.time()

    with SessionLocal() as session:
        session.execute(query)
        session.commit()

    print(f"✅ Updated nearest districts in {time.time() - start:.2f}s")

def update_all_university_locations():
    '''更新所有学校的地理信息'''
    session = SessionLocal()
    try:
        universities = session.query(University).all()

        print(f"Found {len(universities)} universities to update.")

        failcount = 0
        for u in universities:
            name = u.name
            name = re.sub(r'\s*\([^)]*\)', '', name).strip()
            
            lon, lat = get_longitude_latitude(name)
            if lon and lat:
                u.longitude = lon
                u.latitude = lat

                u.geog = from_shape(Point(lon, lat), srid=4326)
                u.geom = from_shape(Point(lon, lat), srid=3414)

                # print(f"✅ Updated {d.neighbour_police_center} ({lat}, {lon})")
            else:
                print(f'❌ Failed to get location for: {u.id} : {name}')
                failcount += 1

            time.sleep(0.3)

        session.commit()
        print("All locations updated successfully.")
        print(f"Failed to update {failcount} records.")
    except Exception as e:
        session.rollback()
        print("Error:", e)
    finally:
        session.close()

def open_map_routing_url(start_lon, start_lat, end_lon, end_lat, mode="TRANSIT"):
    _, _, routing_url, token = get_openmap_token()
    mode = mode
    url = f"{routing_url}start={start_lat}%2C{start_lon}&end={end_lat}%2C{end_lon}&routeType=pt&date=08-13-2025&time=08%3A35%3A00&mode={mode}&numItineraries=3"
    return url, token

def commute_time_between_points(lon1, lat1, lon2, lat2, retries=3):
    for attempt in range(retries):
        try:
            url, token = open_map_routing_url(lon1, lat1, lon2, lat2)
            headers = {"Authorization": token}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"⚠️ HTTP {response.status_code}: {response.text[:100]}")
                raise Exception(f"HTTP {response.status_code}")
            data = response.json()
            itineraries = data.get("plan", {}).get("itineraries", [])
            if itineraries:
                duration_sec = itineraries[0]["duration"]
                return max(1, int(duration_sec / 60))  # 至少1分钟
            return None
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # 等待重试
    return None  # 多次失败返回 None

def calculate_housing_to_university_commute_time():
    '''计算所有房源到最近大学的通勤时间并更新数据库'''
    Base.metadata.create_all(engine)
    session = SessionLocal()
    failed_count = 0
    request_count = 0

    start_time = time.time()
    batch_size = 20  # 每20条提交一次

    try:
        housings = session.query(HousingData).filter(
            HousingData.longitude.isnot(None),
            HousingData.latitude.isnot(None)
        ).all()

        universities = session.query(University).filter(
            University.longitude.isnot(None),
            University.latitude.isnot(None)
        ).all()

        print(f"Calculating commute times for {len(housings)} housings to universities.")

        new_records = []
        for i, housing in enumerate(housings, start=1):
            for uni in universities:
                existing = session.query(CommuteTime).filter_by(
                    housing_id=housing.id, university_id=uni.id
                ).first()
                if existing:
                    continue  # 已存在则跳过

                commute_min = commute_time_between_points(
                    housing.longitude, housing.latitude,
                    uni.longitude, uni.latitude
                )
                request_count += 1

                if commute_min is not None:
                    new_records.append(CommuteTime(
                        housing_id=housing.id,
                        university_id=uni.id,
                        commute_time_minutes=commute_min
                    ))
                    print(f"✅ {housing.name} → {uni.name}: {commute_min} min")
                else:
                    failed_count += 1
                    print(f"❌ Failed for {housing.name} → {uni.name}")

                # 控制速率
                if request_count % 100 == 0:
                    print("⏸️ Paused 2s to avoid rate limiting")
                    time.sleep(2)
                else:
                    time.sleep(0.4)

                # 每20条提交一次
                if len(new_records) >= batch_size:
                    session.add_all(new_records)
                    session.commit()
                    new_records.clear()
                    print(f"💾 Batch committed ({i}/{len(housings)} processed)")

        # 提交剩余的
        if new_records:
            session.add_all(new_records)
            session.commit()

        print(f"✅ All commute times updated successfully in {time.time() - start_time:.2f}s")
        print(f"❗ Failed count: {failed_count}")
    except Exception as e:
        session.rollback()
        print("Error:", e)
    finally:
        session.close()

def get_all_libraries_from_onemap():
    '''从OneMap获取所有图书馆数据'''
    url, token = get_openmap_library_url()
    headers = {"Authorization": token}
    library_response = requests.request("GET", url, headers=headers)
    libraries = []
    data = library_response.json()
    facilities = data.get("SrchResults", [])

    if not facilities:
        print("No library data found from OneMap.")
        return libraries

    meta = facilities[0]
    feat_count = meta.get("FeatCount", 0)

    if feat_count == 0:
        print("No libraries found in the data.")
        return libraries
    
    for item in facilities[1:]:
        name = item.get("NAME")
        latlng = item.get("LatLng")
        if not name or not latlng:
            continue
        
        lat, lon = map(float, latlng.split(","))
        library = Library(name=name, latitude=lat, longitude=lon)
        library.geog = from_shape(Point(lon, lat), srid=4326)
        library.geom = from_shape(Point(lon, lat), srid=3414)
        libraries.append(library)
        print(f"✅ Extracted library: {name} at ({lat}, {lon})")

    print(f"Total libraries extracted: {len(libraries)}")
    return libraries

def insert_all_libraries_to_db():
    '''将所有图书馆数据插入数据库'''
    libraries = get_all_libraries_from_onemap()
    session = SessionLocal()
    try:
        session.add_all(libraries)
        session.commit()
        print(f"✅ Inserted {len(libraries)} libraries into the database.")
    except Exception as e:
        session.rollback()
        print("Error inserting libraries:", e)
    finally:
        session.close()

# 定义要计算的设施类型和对应模型
FACILITY_MODELS = {
    "park": Park,
    "hawkercenter": HawkerCenter,
    "supermarket": Supermarket,
    "library": Library,
}

DATABASE_URL_ASYNC = get_database_url_async()
async_engine = create_async_engine(
    DATABASE_URL_ASYNC, 
    echo=False,)

AsyncSessionLocal = sessionmaker(
    bind=async_engine, class_=AsyncSession, expire_on_commit=False
)

async def compute_nearest_facilities(session: AsyncSession, facility_type: str, model, limit_n: int = 3):
    '''
    为所有房源计算给定设施类型最近的 N 个设施距离
    并 upsert 到 housing_facility_distances 表中
    '''
    print(f"开始计算 {facility_type} 最近的 {limit_n} 个设施...")
    start_time = time.time()

    # 使用纯 SQL 实现高效 lateral join
    sql = text(f"""
        INSERT INTO housing_facility_distances
            (housing_id, facility_type, facility_id, facility_name, rank, distance_m)
        SELECT
            h.id AS housing_id,
            :facility_type AS facility_type,
            f.id AS facility_id,
            f.name AS facility_name,
            ROW_NUMBER() OVER (PARTITION BY h.id ORDER BY h.geog <-> f.geog) AS rank,
            ST_Distance(h.geog, f.geog) AS distance_m
        FROM housing_data h
        JOIN LATERAL (
            SELECT id, name, geog
            FROM {model.__tablename__}
            ORDER BY h.geog <-> {model.__tablename__}.geog
            LIMIT :limit_n
        ) f ON TRUE
        ON CONFLICT (housing_id, facility_type, rank)
        DO UPDATE SET
            facility_id = EXCLUDED.facility_id,
            facility_name = EXCLUDED.facility_name,
            distance_m = EXCLUDED.distance_m;
    """)

    await session.execute(sql, {"facility_type": facility_type, "limit_n": limit_n})
    await session.commit()

    print(f"✅ {facility_type} 计算完成，用时 {time.time() - start_time:.2f}s")

async def run_precompute():
    '''执行预计算'''
    async with AsyncSessionLocal() as session:
        # 确保目标表存在
        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS housing_facility_distances (
                id SERIAL PRIMARY KEY,
                housing_id INTEGER REFERENCES housing_data(id),
                facility_type VARCHAR(50),
                facility_id INTEGER,
                facility_name VARCHAR(255),
                rank INTEGER,
                distance_m FLOAT,
                UNIQUE (housing_id, facility_type, rank)
            );
        """))
        await session.commit()

    # 并发执行 4 个设施类型计算任务（每个任务单独 session）
    async def run_one_type(facility_type, model):
        async with AsyncSessionLocal() as sub_session:
            await compute_nearest_facilities(sub_session, facility_type, model)

    tasks = [
        run_one_type(ftype, fmodel)
        for ftype, fmodel in FACILITY_MODELS.items()
    ]

    await asyncio.gather(*tasks)

    print("所有设施距离预计算完成。")

if __name__ == "__main__":
    # long, lat = get_longitude_latitude("Singapore University of Social Sciences (SUSS)")
    # print(f"Longitude: {long}, Latitude: {lat}")
    # update_all_npc_locations() # 更新警署地理信息
    # update_all_housing_locations() # 更新房源地理信息
    # find_housing_district() # 为房源匹配最近警署
    # update_all_university_locations() # 更新学校地理信息

    # commute_time = commute_time_between_points(103.893713485123, 1.38290495731413, 103.852207943172, 1.29684671398423)
    # print(f"Commute time: {commute_time} minutes")

    # calculate_housing_to_university_commute_time() # 计算所有房源到所有大学的通勤时间
    # get_all_libraries_from_onemap()
    # insert_all_libraries_to_db() # 插入所有图书馆数据
    asyncio.run(run_precompute())