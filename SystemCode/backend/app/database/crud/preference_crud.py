from typing import Optional
from fastapi import Depends
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert as pg_insert 
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.dependencies import get_async_session
from app.models.preference import UserPreference


# 插入模型计算出的用户偏好
async def upsert_user_preference(
    *,
    db: AsyncSession,
    preference: UserPreference
) -> Optional[UserPreference]:
    
    values_to_insert = preference.model_dump(exclude_unset=True) 
    
    statement = pg_insert(UserPreference).values(**values_to_insert)
    
    update_statement = statement.on_conflict_do_update(
        index_elements=['device_id'],
        set_={
            'costScore': statement.excluded.costScore,
            'commuteScore': statement.excluded.commuteScore,
            'neighborhoodScore': statement.excluded.neighborhoodScore,
        }
    ).returning(UserPreference)

    try:
        result = await db.execute(update_statement)
        upserted_preference = result.scalar_one_or_none()
        
        await db.commit() 
        
        if upserted_preference:
            await db.refresh(upserted_preference) 
            print(f"Successfully upserted preference for device {upserted_preference.device_id}.")
            return upserted_preference
        else:
            print(f"Upsert for device {preference.device_id} seemed to succeed but returned no object.")
            return None

    except SQLAlchemyError as e:
        await db.rollback()
        print(f"Database error during upsert preference for device {preference.device_id}. Error: {e}", exc_info=True)
        return None


# 根据device_id查询用户偏好数据
async def get_user_preference_by_device_id(
    *,
    db: AsyncSession,
    device_id: str
) -> Optional[UserPreference]:
    
    preference: Optional[UserPreference] = None
    try:
        statement = select(UserPreference).where(UserPreference.device_id == device_id)
        results = await db.exec(statement)
        preference = results.first()
        
        if preference:
            print(f"Retrieved preference for device {device_id}.")
        else:
            print(f"No preference found for device {device_id}.")

    except SQLAlchemyError as e:
        print(f"Database error occurred while fetching preference for device {device_id}. Error: {e}", exc_info=True)

    return preference
