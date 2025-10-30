from fastapi import APIRouter, Depends, status
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.behavior import UserBehaviorBase
from app.dependencies import get_async_session
from app.handlers import behavior_handler


router = APIRouter(prefix="/api/v1/behaviors", tags=["behaviors"])

# user click behavior
@router.post("/click", status_code=status.HTTP_204_NO_CONTENT)
async def click(
    *,
    db: AsyncSession = Depends(get_async_session),
    behavior: UserBehaviorBase
):
    await behavior_handler.click_behavior_handler(db=db, behavior=behavior)


# user view behavior
@router.post("/view", status_code=status.HTTP_204_NO_CONTENT)
async def view(
    *,
    db: AsyncSession = Depends(get_async_session),
    behavior: UserBehaviorBase
):
    await behavior_handler.view_behavior_handler(db=db, behavior=behavior)


# user favorite behavior
@router.post("/favorite", status_code=status.HTTP_204_NO_CONTENT)
async def favorite(
    *,
    db: AsyncSession = Depends(get_async_session),
    behavior: UserBehaviorBase
):
    await behavior_handler.favorite_behavior_handler(db=db, behavior=behavior)