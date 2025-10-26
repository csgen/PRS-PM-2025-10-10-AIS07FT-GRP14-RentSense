from decimal import Decimal
from typing import Optional

from sqlmodel import Field, SQLModel


# 请求地图模型
class PropertyLocation(SQLModel):
    property_id: int  
    latitude: Optional[Decimal] = Field(default=None)
    longitude: Optional[Decimal] = Field(default=None)


# 返回给前端房源模型 & 算法返回房源模型
class Property(PropertyLocation):
    img_src: Optional[str] = Field(default=None, max_length=500)
    name: Optional[str] = Field(default=None, max_length=100)
    district: Optional[str] = Field(default=None, max_length=100)
    price: Optional[str] = Field(default=None, max_length=50)
    beds: Optional[int] = Field(default=None)
    baths: Optional[int] = Field(default=None)
    area: Optional[int] = Field(default=None)
    build_time: Optional[str] = Field(default=None, max_length=50)
    location: Optional[str] = Field(default=None, max_length=100)
    time_to_school: Optional[int] = Field(default=None)
    distance_to_mrt: Optional[int] = Field(default=None)
    public_facilities: Optional[list[dict]] = Field(default=None)
    facility_type: Optional[str] = Field(default=None, max_length=50)

    # range(0, 1]
    costScore: Optional[float] = Field(default=0.5) 
    commuteScore: Optional[float] = Field(default=0.5) 
    neighborhoodScore: Optional[float] = Field(default=0.5)  

    recommand_reason: Optional[str] = Field(default=None)
