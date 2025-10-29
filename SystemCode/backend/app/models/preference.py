from sqlmodel import Field, SQLModel


# 用户偏好权重模型
# 由用户行为数据训练的时序模型训练得到
class UserPreference(SQLModel, table=True):
    __tablename__ = "ppp"

    device_id: str = Field(max_length=100, primary_key=True)

    # range(0, 1]
    costScore: float = Field(default=0.5) 
    commuteScore: float = Field(default=0.5) 
    neighborhoodScore: float = Field(default=0.5)  
