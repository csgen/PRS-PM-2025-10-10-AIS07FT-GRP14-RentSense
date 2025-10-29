import json
from typing import List, Optional

from sqlmodel import select
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


async def get_recommendation(
    *,
    eid: int,
    db: AsyncSession
) -> Optional[Recommendation]:
    
    if not eid:
        print('Failed to retrieve recommendation. Error: eid is None.')
        return None
    
    cache_key = f"recommendation:{eid}"
    recommendation: Optional[Recommendation] = None

    # 尝试从 Redis 缓存中获取
    try:
        cached_result = await redis_client.get(cache_key)
        
        if cached_result:
            try:
                recommendation = Recommendation.model_validate_json(cached_result)
                print(f"Successfully retrieved recommendation for enquiry {eid} from Redis.")
                return recommendation
                
            except (json.JSONDecodeError, TypeError) as e:
                await redis_client.delete(cache_key)
                print(f"Corrupted cache data for enquiry {eid}. Deleting cache. Error: {e}")

    except RedisError as e:
        print(f"Failed to connect to Redis for enquiry {eid}. Proceeding to database. Error: {e}")


    # 2. 缓存未命中或缓存失败，从数据库读取
    try:
        statement = select(Recommendation).where(Recommendation.eid == eid).order_by(Recommendation.rid.desc())
        
        result = await db.exec(statement)
        recommendation = result.first()
        
        if recommendation:
            print(f"Successfully retrieved recommendation {recommendation.rid} for enquiry {eid} from database.")
            
            # 3. 数据库读取成功，更新缓存
            try:
                recommendation_json = recommendation.model_dump_json()
                await redis_client.set(cache_key, recommendation_json, ex=CACHE_TTL_SECONDS)
                print(f"Successfully updated Redis cache for enquiry {eid}.")
                
            except (RedisError, TypeError) as e:
                print(f"Failed to update cache for enquiry {eid}. Error: {e}")

        else:
            print(f"No recommendation found for enquiry {eid} in database.")

    except SQLAlchemyError as e:
        print(f"Failed to retrieve recommendation for enquiry {eid} from database. Error: {e}")

    return recommendation
