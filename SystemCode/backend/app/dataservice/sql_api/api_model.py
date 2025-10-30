from pydantic import BaseModel, Field, conint, confloat
from typing import Optional

class RequestInfo(BaseModel):
    device_id: Optional[str] = Field(default=None, max_length=100, index=True)
    min_monthly_rent: int = Field(..., description="最低月租 (S$)")
    max_monthly_rent: int = Field(..., description="最高月租 (S$)")
    school_id: conint(ge=1, le=6) = Field(..., description="目标学校id（1-6）")

    # 可选字段（可能为空）
    target_district_id: Optional[conint(ge=1, le=36)] = Field(None, description="目标区域id（id：1-36）")
    max_school_limit: Optional[int] = Field(None, description="最远能接受的去学校的时间（min）")
    flat_type_preference: Optional[list[str]] = Field(default_factory=list, description="房源类型名称（flat_type）")
    max_mrt_distance: Optional[int] = Field(None, description="最远能接受的离最近的MRT距离")
    importance_rent: Optional[conint(ge=1, le=5)] = Field(None, description="房租重要程度 (1-5)")
    importance_location: Optional[conint(ge=1, le=5)] = Field(None, description="位置重要程度(1-5)")
    importance_facility: Optional[conint(ge=1, le=5)] = Field(None, description="便利重要程度（1-5）")

class ResultInfo(BaseModel):
    property_id: int = Field(..., description="房源编号")
    img_src: str = Field(..., description="房源图片地址")
    name: str = Field(..., description="名称")
    district: str = Field(..., description="房源所属区域名称")
    price: str = Field(..., description="价格")
    beds: int
    baths: int
    area: int = Field(..., description="面积")
    build_time: str = Field(default="", description="建造时间(如果没有，返回空字符串)")
    location: str = Field(..., description="地址信息")
    time_to_school: int = Field(..., description="距离指定学校的距离")
    distance_to_mrt: int = Field(..., description="距离最近mrt距离")
    latitude: float
    longitude: float
    public_facilities: list[dict[str, str]] = Field(default_factory=list, description="附近设施（名称，距离m）")
    facility_type: str = Field(..., description="房源类型 hdb/condo等")
    costScore: confloat(ge=0, le=1) = Field(..., description="成本评分 range(0, 1]")
    commuteScore: confloat(ge=0, le=1) = Field(..., description="通勤评分 range(0, 1]")
    neighborhoodScore: confloat(ge=0, le=1) = Field(..., description="设施及安全的综合评分 range(0, 1]")