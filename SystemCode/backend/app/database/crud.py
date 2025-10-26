from typing import List, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel.ext.asyncio.session import AsyncSession
from redis import RedisError

from app.models import EnquiryForm, EnquiryEntity, Property, Recommendation
from app.database.cache import redis_client, CACHE_TTL_SECONDS


async def save_enquiry(
    *,
    db: AsyncSession,
    enquiry: EnquiryForm
) -> Optional[EnquiryEntity]:
    
    enquiry_entity = EnquiryEntity.model_validate(enquiry)
    saved_entity: Optional[EnquiryEntity] = None

    try:
        # db
        db.add(enquiry_entity)
        await db.commit()
        await db.refresh(enquiry_entity)
        saved_entity = enquiry_entity
        print(f"Successfully saved enquiry {enquiry_entity.eid} to database.")

        # cache
        if saved_entity and saved_entity.eid:
            try:
                cache_key = f"enquiry:{enquiry_entity.eid}"
                enquiry_json = enquiry_entity.model_dump_json()
                await redis_client.set(cache_key, enquiry_json, ex=CACHE_TTL_SECONDS)
                print(f"Successfully cached enquiry {enquiry_entity.eid} to Redis.")

            except RedisError as e:
                print(f"Failed to cache enquiry {enquiry_entity.eid} to Redis. Error: {e}")
        
        else:
            print("Eid is missing. Cannot cache.")
    
    except SQLAlchemyError as e:
        await db.rollback()
        print(f"Failed to save enquiry to database. Error: {e}")
    
    return saved_entity


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
