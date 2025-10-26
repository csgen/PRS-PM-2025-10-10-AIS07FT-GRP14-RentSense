from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from pydantic import computed_field
from sqlalchemy import JSON, Column, DateTime, func
from sqlmodel import Field, Relationship, SQLModel

from .property import Property
if TYPE_CHECKING:
    from .enquiry import EnquiryEntity


# 存到数据库的推荐结果模型
class Recommendation(SQLModel, table=True):
    __tablename__ = "recommendations"

    rid: Optional[int] = Field(default=None, primary_key=True)
    eid: int = Field(foreign_key="enquiries.eid", unique=True)
    
    create_time: datetime = Field(
        default_factory=datetime.utcnow, 
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    )
    
    recommandation_result: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))
    
    ext_info: Optional[dict] = Field(default=None, sa_column=Column(JSON))

    enquiry: "EnquiryEntity" = Relationship(back_populates="recommendation")


# 返回给前端的推荐结果模型
class RecommendationResponse(SQLModel):
    properties: List[Property]

    @computed_field
    @property
    def total_count(self) -> int:
        return len(self.properties)
