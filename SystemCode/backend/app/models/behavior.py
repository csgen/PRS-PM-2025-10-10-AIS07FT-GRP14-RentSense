from typing import Optional
from datetime import datetime

from sqlalchemy import Column, DateTime, func
from sqlmodel import Field, SQLModel

from app.dataservice.sql_api.api_model import ResultInfo


class UserBehaviorBase(SQLModel):
    device_id: Optional[str] = Field(default=None, max_length=100, index=True)
    property_id: int  
    duration: Optional[float] = Field(default=0.0)
    favorite: bool = Field(default=False)
    
    update_time: datetime = Field(
        default_factory=datetime.utcnow
    )


# 用户行为模型
class UserBehavior(UserBehaviorBase, table=True):
    __tablename__ = "behaviors"

    bid: Optional[int] = Field(default=None, primary_key=True)
    
    update_time: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    )


# 用户行为完整数据，包含对应房源的详细信息，用于推荐模型训练
class UserBehaviorComplete(UserBehavior, ResultInfo):
    pass

