from sqlalchemy import Column, Integer, BigInteger, String, Float, Boolean, Text, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from geoalchemy2 import Geometry, Geography

Base = declarative_base()

class HousingData(Base):
    '''房源数据'''
    __tablename__ = 'housing_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(500))
    price = Column(Integer)
    area_sqft = Column(Integer)
    build_time = Column(Integer, nullable=True)
    type = Column(String(100))
    location = Column(Text)
    distance_to_mrt = Column(Integer)
    availability = Column(Text)
    beds_num = Column(Integer)
    baths_num = Column(Integer)
    is_room = Column(Boolean)
    district_id = Column(Integer, nullable=True)
    longitude = Column(Float, nullable=True)
    latitude = Column(Float, nullable=True)

    geom = Column(Geometry(geometry_type='POINT', srid=3414), nullable=True)
    geog = Column(Geography(geometry_type='POINT', srid=4326), nullable=True)

class District(Base):
    '''按警署划分的区域'''
    __tablename__ = 'districts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    neighbour_police_center = Column(String(255))
    district_name = Column(String(255))
    num_in_2024 = Column(Integer, nullable=True)
    num_in_2023 = Column(Integer, nullable=True)
    num_in_2022 = Column(Integer, nullable=True)
    num_in_2021 = Column(Integer, nullable=True)
    num_in_2020 = Column(Integer, nullable=True)
    average_num = Column(Float)
    safety_score = Column(Float)
    postal_code = Column(String(20))
    longitude = Column(Float, nullable=True)
    latitude = Column(Float, nullable=True)

    geom = Column(Geometry(geometry_type='POINT', srid=3414), nullable=True)
    geog = Column(Geography(geometry_type='POINT', srid=4326), nullable=True)

class University(Base):
    '''学校数据'''
    __tablename__ = 'universities'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    longitude = Column(Float, nullable=True)
    latitude = Column(Float, nullable=True)

    geom = Column(Geometry(geometry_type='POINT', srid=3414), nullable=True)
    geog = Column(Geography(geometry_type='POINT', srid=4326), nullable=True)

class CommuteTime(Base):
    '''房源 - 大学 通勤时间表'''
    __tablename__ = "commute_times"

    id = Column(Integer, primary_key=True, autoincrement=True)
    housing_id = Column(Integer, ForeignKey("housing_data.id"), nullable=False)
    university_id = Column(Integer, ForeignKey("universities.id"), nullable=False)
    commute_time_minutes = Column(Float, nullable=True)  # 单位：分钟

    # 防止重复记录
    __table_args__ = (UniqueConstraint('housing_id', 'university_id', name='_housing_university_uc'),)

    # ORM 关系
    housing = relationship("HousingData", backref="commute_records")
    university = relationship("University", backref="commute_records")

class Park(Base):
    '''公园数据'''
    __tablename__ = 'parks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    longitude = Column(Float, nullable=True)
    latitude = Column(Float, nullable=True)

    geom = Column(Geometry(geometry_type='POINT', srid=3414), nullable=True)
    geog = Column(Geography(geometry_type='POINT', srid=4326), nullable=True)

class HawkerCenter(Base):
    '''食阁数据'''
    __tablename__ = 'hawker_centers'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    longitude = Column(Float, nullable=True)
    latitude = Column(Float, nullable=True)

    geom = Column(Geometry(geometry_type='POINT', srid=3414), nullable=True)
    geog = Column(Geography(geometry_type='POINT', srid=4326), nullable=True)

class Supermarket(Base):
    '''超市数据'''
    __tablename__ = 'supermarkets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    longitude = Column(Float, nullable=True)
    latitude = Column(Float, nullable=True)

    geom = Column(Geometry(geometry_type='POINT', srid=3414), nullable=True)
    geog = Column(Geography(geometry_type='POINT', srid=4326), nullable=True)

class Library(Base):
    '''图书馆数据'''
    __tablename__ = 'libraries'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    longitude = Column(Float, nullable=True)
    latitude = Column(Float, nullable=True)

    geom = Column(Geometry(geometry_type='POINT', srid=3414), nullable=True)
    geog = Column(Geography(geometry_type='POINT', srid=4326), nullable=True)

class HousingFacilityDistance(Base):
    __tablename__ = 'housing_facility_distances'

    id = Column(Integer, primary_key=True, autoincrement=True)
    housing_id = Column(Integer, ForeignKey('housing_data.id'), index=True)
    facility_type = Column(String(50))  # 'park', 'hawker', 'library', 'supermarket'
    facility_id = Column(Integer)
    facility_name = Column(String(255))
    rank = Column(Integer)  # 第几个最近的（1, 2, 3）
    distance_m = Column(Float)

class ImageRecord(Base):
    '''图片数据'''
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False, index=True)
    gcs_url = Column(Text, nullable=False)
    public_url = Column(Text, nullable=False)
    file_size = Column(BigInteger)
    content_type = Column(String(100))
    bucket_name = Column(String(255))
    gcs_path = Column(String(500))
    
    def __repr__(self):
        return f"<ImageRecord(id={self.id}, filename='{self.filename}', public_url='{self.public_url}')>"