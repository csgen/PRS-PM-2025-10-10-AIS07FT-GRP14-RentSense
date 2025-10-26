import pandas as pd
from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Float, Boolean, Text
from sqlalchemy.orm import sessionmaker
import sys, os
import numpy as np
import json
from geoalchemy2.shape import from_shape
from shapely.geometry import Point
from bs4 import BeautifulSoup
current_dir = os.path.dirname(__file__) 
project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) 
sys.path.insert(0, project_root) 
from SystemCode.db.envconfig import get_database_url
from SystemCode.db.model import Base, HousingData, District, University, Park, HawkerCenter, Supermarket

def upload_housing_data(csv_file_path, database_url):
    """
    上传住房数据到PostgreSQL数据库
    """
    try:
        # 创建数据库引擎
        engine = create_engine(database_url)
        
        # 先创建表结构（这会创建包含自增主键id的表）
        Base.metadata.create_all(engine)
        
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        
        # 数据清洗和转换
        # 1. 删除不需要的列
        columns_to_drop = ['Beds', 'Baths']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # 2. 处理Is_room列：将不是Yes的值转换为No（在数据库中为False）
        if 'Is_room' in df.columns:
            df['Is_room'] = df['Is_room'].apply(lambda x: True if str(x).strip().lower() == 'yes' else False)
        
        # 3. 重命名列以匹配数据库表结构
        column_mapping = {
            'Name': 'name',
            'Price': 'price',
            'Area(sqft)': 'area_sqft',
            'Build_Time': 'build_time',
            'Type': 'type',
            'Location': 'location',
            'Distance_to_MRT': 'distance_to_mrt',
            'Availability': 'availability',
            'Beds_num': 'beds_num',
            'Baths_num': 'baths_num',
            'Is_room': 'is_room',
            'District_id': 'district_id',
            'Longitude': 'longitude',
            'Latitude': 'latitude'
        }
        df = df.rename(columns=column_mapping)

        
        # 4. 处理空值
        df = df.replace({'—': None, '': None, ' ': None, '-': None, 'na': None, 'NA': None})

        # 建筑年份字段 - 处理特殊字符并确保合理范围
        if 'build_time' in df.columns:
            # 先转换为字符串，然后清理
            df['build_time'] = df['build_time'].astype(str)
            # 移除非数字字符
            df['build_time'] = df['build_time'].str.replace(r'[^\d]', '', regex=True)
            # 转换为数值，无法转换的设为NaN
            df['build_time'] = pd.to_numeric(df['build_time'], errors='coerce')
            # 确保年份在合理范围内 (1900-2024)
            df['build_time'] = df['build_time'].apply(
                lambda x: x if 1900 <= x <= 2024 else None
            )

        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            records = []
            for _, row in df.iterrows():
                district = HousingData(
                    name=row['name'],
                    price=int(row['price']) if not pd.isna(row['price']) else None,
                    area_sqft=int(row['area_sqft']) if not pd.isna(row['area_sqft']) else None,
                    build_time=int(row['build_time']) if not pd.isna(row['build_time']) else None,
                    type=row['type'],
                    location=row['location'],
                    distance_to_mrt=row['distance_to_mrt'],
                    availability=row['availability'],
                    beds_num=row['beds_num'],
                    baths_num=row['baths_num'],
                    is_room=row['is_room'],
                    district_id=int(row['district_id']) if not pd.isna(row['district_id']) else None,
                    longitude=float(row['longitude']) if not pd.isna(row['longitude']) else None,
                    latitude=float(row['latitude']) if not pd.isna(row['latitude']) else None
                )
                records.append(district)
            
            session.add_all(records)
            session.commit()
            print(f"成功上传 {len(records)} 条房源数据到数据库")
            
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
        
    except Exception as e:
        print(f"上传过程中出现错误: {str(e)}")
        raise

def upload_district_data(csv_file_path, database_url):
    """
    上传区域数据到PostgreSQL数据库
    """
    try:
        # 创建数据库引擎
        engine = create_engine(database_url)
        
        # 先创建表结构（这会创建包含自增主键id的表）
        Base.metadata.create_all(engine)
        
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)

        # 数据清洗步骤
        year_columns = ['2024', '2023', '2022', '2021', '2020']
        for col in year_columns:
            df[col] = df[col].replace('na', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.replace({'—': None, '': None, ' ': None})
        df['Postcode'] = df['Postcode'].astype(str)
        
        # 重命名列
        column_mapping = {
            'Neighbour Police Center': 'neighbour_police_center',
            'District Name': 'district_name',
            '2024': 'num_in_2024',
            '2023': 'num_in_2023',
            '2022': 'num_in_2022',
            '2021': 'num_in_2021',
            '2020': 'num_in_2020',
            'Average': 'average_num',
            'safety_score': 'safety_score',
            'Postcode': 'postal_code',
            'Longitude': 'longitude',
            'Latitude': 'latitude'
        }
        df = df.rename(columns=column_mapping)
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            records = []
            for _, row in df.iterrows():
                district = District(
                    neighbour_police_center=row['neighbour_police_center'],
                    district_name=row['district_name'],
                    num_in_2024=row['num_in_2024'] if not pd.isna(row['num_in_2024']) else None,
                    num_in_2023=row['num_in_2023'] if not pd.isna(row['num_in_2023']) else None,
                    num_in_2022=row['num_in_2022'] if not pd.isna(row['num_in_2022']) else None,
                    num_in_2021=row['num_in_2021'] if not pd.isna(row['num_in_2021']) else None,
                    num_in_2020=row['num_in_2020'] if not pd.isna(row['num_in_2020']) else None,
                    average_num=row['average_num'],
                    safety_score=row['safety_score'],
                    postal_code=row['postal_code'],
                    longitude=row['longitude'],
                    latitude=row['latitude']
                )
                records.append(district)
            
            session.add_all(records)
            session.commit()
            print(f"成功上传 {len(records)} 条区域数据到数据库")
            
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
        
    except Exception as e:
        print(f"上传过程中出现错误: {str(e)}")
        raise

university_names = [
    "National University of Singapore (NUS)",
    "Nanyang Technological University (NTU)", 
    "Singapore Management University (SMU)",
    "Singapore University of Technology and Design (SUTD)",
    "Singapore Institute of Technology (SIT)",
    "Singapore University of Social Sciences (SUSS)"
]

def insert_all_universities(database_url):
    # 创建数据库引擎
    engine = create_engine(database_url)
    
    # 创建表结构
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        # 大学列表
        universities = []
        for name in university_names:
            university = University(name=name)
            universities.append(university)
        
        # 批量插入
        session.add_all(universities)
        session.commit()
        print(f"成功插入 {len(universities)} 所大学")
        
    except Exception as e:
        session.rollback()
        print(f"插入数据时出错: {e}")
    finally:
        session.close()

def extract_from_parks() -> list[Park]:
    '''从GeoJSON文件中提取公园数据'''
    filepath = '..\..\Miscellaneous\Parks.geojson'
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    failed_count = 0
    success_count = 0
    parks = []
    for feature in data['features']:
        try:
            # 提取数据
            properties = feature.get('properties', {})
            geometry = feature.get('geometry', {})
            
            # 获取公园名称
            name = properties.get('NAME', '').strip()
            if not name:
                print(f"跳过没有名称的特征: {feature.get('id', '未知ID')}")
                failed_count += 1
                continue
            
            # 获取坐标
            if geometry.get('type') == 'Point':
                coordinates = geometry.get('coordinates', [])
                if len(coordinates) >= 2:
                    longitude = float(coordinates[0])  # 经度
                    latitude = float(coordinates[1])   # 纬度
                else:
                    print(f"跳过坐标不完整的特征: {name}")
                    failed_count += 1
                    continue
            else:
                print(f"跳过非点类型的特征: {name}")
                failed_count += 1
                continue
            # print(f"公园: {name} - 经度: {longitude}, 纬度: {latitude}")
            # 创建公园对象
            park = Park(
                name=name,
                longitude=longitude,
                latitude=latitude
            )
            park.geog = from_shape(Point(longitude, latitude), srid=4326)
            park.geom = from_shape(Point(longitude, latitude), srid=3414)
            parks.append(park)
            success_count += 1
        except Exception as e:
                print(f"处理特征时出错 {feature.get('id', '未知ID')}: {str(e)}")
                failed_count += 1
                continue
    print(f"提取完成，成功: {success_count}，失败: {failed_count}")
    return parks

def insert_all_parks(database_url):
    '''插入所有公园数据到数据库'''
    # 创建数据库引擎
    engine = create_engine(database_url)
    
    # 创建表结构
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    inserted_count = 0

    try:
        parks = extract_from_parks()
        session.add_all(parks)
        session.commit()
        inserted_count = len(parks)
        print(f"成功插入 {inserted_count} 个公园数据")
        
    except Exception as e:
        session.rollback()
        print(f"插入数据时出错: {str(e)}")
        raise
    finally:
        session.close()

def extract_from_hawkercenters() -> list[HawkerCenter]:
    '''从GeoJSON文件中提取食阁数据'''
    filepath = '..\..\Miscellaneous\HawkerCentresGEOJSON.geojson'
    with open(filepath, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    
    features = geojson_data.get('features', [])
    print(f"{len(features)} 个hawkercenter记录")
    hawker_centers = []
    failed_count = 0
    success_count = 0
    for feature in features:
        try:
            geometry = feature.get('geometry', {})
            coordinates = geometry.get('coordinates', [])
            
            if len(coordinates) != 2:
                print(f"跳过无效坐标: {coordinates}")
                failed_count += 1
                continue
            
            longitude = coordinates[0]
            latitude = coordinates[1]
            
            # 提取properties中的name
            properties = feature.get('properties', {})
            name = properties.get('NAME', '')
            
            if not name:
                print("跳过无名记录")
                failed_count += 1
                continue
            
            # 创建HawkerCenter对象
            hawker_center = HawkerCenter(
                name=name,
                longitude=longitude,
                latitude=latitude
            )
            print(f"✅ 提取: hawkercenter:{name} lat: {latitude}, lon: {longitude}")
            success_count += 1
            # 设置geom和geog字段
            hawker_center.geog = from_shape(Point(longitude, latitude), srid=4326)
            hawker_center.geom = from_shape(Point(longitude, latitude), srid=3414)
            hawker_centers.append(hawker_center)
        except Exception as e:
            print(f"处理记录时出错: {e}")
            failed_count += 1
            continue
    print(f"提取完成，成功: {success_count}，失败: {failed_count}")
    return hawker_centers

def insert_all_hawkercenters(database_url):
    '''插入所有食阁数据到数据库'''
    engine = create_engine(database_url)
    
    # 创建表结构
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    inserted_count = 0
    try:
        hwkcenters = extract_from_hawkercenters()
        session.add_all(hwkcenters)
        session.commit()
        inserted_count = len(hwkcenters)
        print(f"成功插入 {inserted_count} 个食阁数据")
        
    except Exception as e:
        session.rollback()
        print(f"插入数据时出错: {str(e)}")
        raise
    finally:
        session.close()

def extract_name_from_description(description_html):
    """
    从HTML描述中提取超市名称
    
    Args:
        description_html (str): HTML格式的描述文本
        
    Returns:
        str: 超市名称，如果提取失败返回空字符串
    """
    try:
        soup = BeautifulSoup(description_html, 'html.parser')
        
        # 查找包含LIC_NAME的行
        rows = soup.find_all('tr')
        for row in rows:
            th = row.find('th')
            if th and 'LIC_NAME' in th.get_text():
                td = row.find('td')
                if td:
                    return td.get_text(strip=True)
        
        return ""
    except Exception as e:
        print(f"解析HTML描述时出错: {e}")
        return ""

def extract_from_supermarkets():
    '''从GeoJSON文件中提取超市数据'''

    filepath = '..\..\Miscellaneous\SupermarketsGEOJSON.geojson'
    supermarkets = []
    
    failed_count = 0
    success_count = 0
    with open(filepath, 'r', encoding='utf-8') as file:
        geojson_data = json.load(file)
    
    features = geojson_data.get('features', [])
    
    for feature in features:
        try:
            properties = feature.get('properties', {})
            geometry = feature.get('geometry', {})
            
            # 从Description HTML中提取LIC_NAME
            description_html = properties.get('Description', '')
            name = extract_name_from_description(description_html)
            
            # 从geometry中提取坐标
            coordinates = geometry.get('coordinates', [])
            if len(coordinates) >= 2:
                longitude = float(coordinates[0])
                latitude = float(coordinates[1])
                
                # 创建数据对象
                supermarket = Supermarket(
                    name=name,
                    longitude=longitude,
                    latitude=latitude
                )
                supermarket.geog = from_shape(Point(longitude, latitude), srid=4326)
                supermarket.geom = from_shape(Point(longitude, latitude), srid=3414)
                supermarkets.append(supermarket)
                success_count += 1
                print(f"✅ 提取: supermarket:{name} lat: {latitude}, lon: {longitude}")
                
        except Exception as e:
            print(f"提取数据时出错: {e}")
            failed_count += 1
            continue
    print(f"提取完成，成功: {success_count}，失败: {failed_count}")
    return supermarkets

def insert_all_supermarkets(database_url):
    '''插入所有超市数据到数据库'''
    engine = create_engine(database_url)
    
    # 创建表结构
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    inserted_count = 0
    try:
        supermarkets = extract_from_supermarkets()
        session.add_all(supermarkets)
        session.commit()
        inserted_count = len(supermarkets)
        print(f"成功插入 {inserted_count} 个超市数据")
        
    except Exception as e:
        session.rollback()
        print(f"插入数据时出错: {str(e)}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    db_url = get_database_url()

    # housing_csv_file = "..\..\Miscellaneous\cleaned_housing_data_20251018.csv"
    # upload_housing_data(housing_csv_file, db_url)

    # district_csv_file = "..\..\Miscellaneous\cleaned_crime_data.csv"
    # upload_district_data(district_csv_file, db_url)
    # insert_all_universities(db_url)

    # extract_from_hawkercenters()
    # extract_from_supermarkets()
    # insert_all_parks(db_url)
    # insert_all_hawkercenters(db_url)
    insert_all_supermarkets(db_url)