from typing import Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel.ext.asyncio.session import AsyncSession
from redis import RedisError

from app.models import EnquiryForm, EnquiryEntity
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