from typing import Optional, List
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.preference import UserPreference
from app.models.property import Property
from app.models.behavior import UserBehavior, UserBehaviorBase, UserBehaviorComplete
from app.database.crud import behavior_crud, preference_crud
from app.database import cache
from app.services.user_modeling.user_behavior_predict import predict_user_omega

# new feature
async def click_behavior_handler(
    *,
    db: AsyncSession,
    behavior: UserBehaviorBase
) -> Optional[UserBehavior] :
    await _save_user_behavior(db=db, behavior=behavior)


async def view_behavior_handler(
    *,
    db: AsyncSession,
    behavior: UserBehaviorBase
) -> Optional[UserBehavior] :
    await _save_user_behavior(db=db, behavior=behavior)

    # use behavior data to update user preference
    prop: Optional[Property] = await cache.get_latest_property_by_id(
        property_id=behavior.property_id if behavior else None
    )
    behaviorComplete = _get_behavior_complete(behavior=behavior, prop=prop)

    print(f'==============behaviorComplete: {behaviorComplete.model_dump_json(indent=2)}==============')

    preference: Optional[UserPreference] = predict_user_omega(behaviorComplete)

    print(f'==============preference: {preference.model_dump_json(indent=2)}==============')
    
    preference_crud.upsert_user_preference(db=db, preference=preference)
    


async def favorite_behavior_handler(
    *,
    db: AsyncSession,
    behavior: UserBehaviorBase
) -> Optional[UserBehavior] :
    await _save_user_behavior(db=db, behavior=behavior)


async def _save_user_behavior(
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


def _get_behavior_complete(
    *,
    behavior: UserBehaviorBase, 
    prop: Property
) -> Optional[UserBehaviorComplete]:
    
    if not behavior or not prop or behavior.property_id != prop.property_id:
        return None
    
    prop_data = prop.model_dump()
    prop_data.pop("property_id", None)

    return UserBehaviorComplete(
        **behavior.model_dump(), 
        **prop_data
    )
