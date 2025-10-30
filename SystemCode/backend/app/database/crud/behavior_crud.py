from typing import List, Optional
from datetime import datetime

from sqlmodel import select
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.behavior import UserBehaviorBase, UserBehavior


# 更新用户行为数据
async def upsert_user_behavior(
    *,
    db: AsyncSession,
    behavior: UserBehaviorBase
) -> Optional[UserBehavior]:
    
    statement = select(UserBehavior).where(
        UserBehavior.device_id == behavior.device_id,
        UserBehavior.property_id == behavior.property_id
    )

    existing_behavior: Optional[UserBehavior] = None
    saved_behavior: Optional[UserBehavior] = None

    try:
        results = await db.exec(statement)
        existing_behavior = results.first()

        # 已存在数据，更新
        if existing_behavior:
            print(f"Updating existing behavior for device {behavior.device_id}, property {behavior.property_id}")

            update_data = behavior.model_dump(exclude_unset=True)

            update_data.pop("device_id", None)
            update_data.pop("property_id", None)

            for key, value in update_data.items():
                setattr(existing_behavior, key, value)

            saved_behavior = existing_behavior

        # 不存在数据，插入
        else:
            print(f"Inserting new behavior for device {behavior.device_id}, property {behavior.property_id}")
            new_behavior = UserBehavior.model_validate(behavior)
            db.add(new_behavior)
            saved_behavior = new_behavior
        
        await db.commit()

        if saved_behavior:
            await db.refresh(saved_behavior)
            print(f"Successfully upserted behavior, bid: {saved_behavior.bid}")
             
    except IntegrityError as e:
        await db.rollback()
        print(f"IntegrityError")
    
    except SQLAlchemyError as e:
        await db.rollback()
        print(f"SQLAlchemyError")
    
    return saved_behavior 


# 查询behaviors表中update_time在指定时间之后的所有记录
async def get_user_behaviors_since(
    *,
    db: AsyncSession,
    since: datetime # 起始时间点
) -> List[UserBehaviorBase]:
    
    behaviors_result: List[UserBehaviorBase] = []

    try:
        statement = select(UserBehavior).where(UserBehavior.update_time > since)
        
        results = await db.exec(statement)
        all_behaviors: List[UserBehavior] = results.all()
        
        behaviors_result = [
            UserBehaviorBase.model_validate(behavior) for behavior in all_behaviors
        ]
        
        print(f"Retrieved {len(behaviors_result)} behaviors updated since {since}.")

    except SQLAlchemyError as e:
        print(f"Database error occurred while fetching behaviors since {since}. Error: {e}", exc_info=True)
        
    return behaviors_result
