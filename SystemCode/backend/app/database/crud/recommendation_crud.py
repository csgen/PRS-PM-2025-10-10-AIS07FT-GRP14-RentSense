from typing import List, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel.ext.asyncio.session import AsyncSession
from redis import RedisError

from app.models import Property, Recommendation
from app.database.cache import redis_client, CACHE_TTL_SECONDS


async def save_recommendation(
    *,
    eid: int,
    db: AsyncSession,
    properties: List[Property]
)-> Optional[Recommendation]:
    
    if not eid:
        print('Failed to save recommendation. Error: eid is None.')
        return None
    
    properties_data = [prop.model_dump(mode='json') for prop in properties]
    recommendation = Recommendation(
        eid=eid,
        recommandation_result=properties_data
    )
    saved_recommendation: Optional[Recommendation] = None

    try:
        # db
        db.add(recommendation)
        await db.commit()
        await db.refresh(recommendation)
        saved_recommendation = recommendation
        print(f"Successfully saved recommendation {recommendation.rid} for enquiry {eid} to database.")

        # cache
        if saved_recommendation and saved_recommendation.rid:
            try:
                cache_key = f"recommendation:{eid}"
                recommendation_json = saved_recommendation.model_dump_json()
                await redis_client.set(cache_key, recommendation_json, ex=CACHE_TTL_SECONDS)
                print(f"Successfully saved recommendation {saved_recommendation.rid} for enquiry {eid} to Redis.")

            except (RedisError, TypeError) as e:
                print(f"Failed to cache recommendation object for enquiry {eid}. Error: {e}")

        else:
            print("Rid missing. Cannot cache.")
    
    except SQLAlchemyError as e:
        await db.rollback()
        print(f"Failed to save recommendation for enquiry {eid}. Error: {e}")
    
    return saved_recommendation