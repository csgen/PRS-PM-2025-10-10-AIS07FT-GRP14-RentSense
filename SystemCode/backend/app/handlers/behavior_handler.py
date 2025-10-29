from typing import Optional
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.behavior import UserBehavior, UserBehaviorBase
from app.database.crud import behavior_crud

async def save_user_behavior(
    *,
    db: AsyncSession,
    behavior: UserBehaviorBase
) -> Optional[UserBehavior] :
    
    saved_behavior = await behavior_crud.upsert_user_behavior(db=db, behavior=behavior)

    if saved_behavior:
        print(saved_behavior.model_dump_json(indent=2))
        print(f"=============Successfully saved user behavior! \
              device_id: {behavior.device_id}\
              property_id: {behavior.property_id}=============")
    else:
        print(f"=============Failed to save user behavior, \
              device_id: {behavior.device_id}\
              property_id: {behavior.property_id}=============")
    
    return saved_behavior
