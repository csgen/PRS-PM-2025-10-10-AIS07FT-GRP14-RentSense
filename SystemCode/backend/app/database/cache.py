import os, json
import redis.asyncio as redis
from typing import Optional, List
from redis import RedisError

from app.models import Property


CACHE_TTL_SECONDS = 60 * 60 * 24 # 24 hours in seconds

redis_host = os.getenv("REDIS_HOST")
if not redis_host:
    raise ValueError("REDIS_HOST environment variable is not set.")
redis_port = int(os.getenv("REDIS_PORT", 6379))

_redis_pool = redis.ConnectionPool(
    host=redis_host, 
    port=redis_port, 
    db=0, 
    decode_responses=True
)

redis_client = redis.Redis(connection_pool=_redis_pool)


LAST_RECOMMENDATION_HASH_KEY = "latest_recommendation_hash"


async def save_latest_properties_as_hash(
    *,
    properties: List[Property]
) -> bool:

    if not properties:
        print('Failed to cache properties. Error: Property list is empty.')
        return False
    
    try:
        properties_map = {
            prop.property_id: prop.model_dump_json()
            for prop in properties
            if prop and prop.property_id
        }

        for prop in properties:
            print(f"===============prop: {prop.model_dump_json(indent=2)}")
    
    except Exception as e:
        print(f"Failed to serialize properties into map. Error: {e}")
        return False

    if not properties_map:
        print("No valid properties to map.")
        return False

    try:
        pipe = redis_client.pipeline()
        await pipe.delete(LAST_RECOMMENDATION_HASH_KEY)
        await pipe.hset(LAST_RECOMMENDATION_HASH_KEY, mapping=properties_map)
        await pipe.expire(LAST_RECOMMENDATION_HASH_KEY, CACHE_TTL_SECONDS)
        await pipe.execute()
        print(f"Successfully saved LATEST {len(properties_map)} properties as Hash to Redis (24h TTL).")
        return True

    except (RedisError, TypeError) as e:
        print(f"Failed to cache LATEST properties as Hash. Error: {e}")
        return False


# 根据房源id获取房源详细信息（仅支持最近一次推荐结果中的房源）
async def get_latest_property_by_id(
    *,
    property_id: str
) -> Optional[Property]:
    
    if not property_id:
        return None

    try:
        cached_result = await redis_client.hget(LAST_RECOMMENDATION_HASH_KEY, property_id)
        
        if cached_result:
            try:
                property_obj = Property.model_validate_json(cached_result)
                print(f"Successfully retrieved property {property_id} from Redis Hash.")
                return property_obj
                
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Corrupted cache data for property {property_id}. Error: {e}")
                return None

    except RedisError as e:
        print(f"Failed to connect to Redis while retrieving property {property_id}. Error: {e}")
    
    print(f"Property {property_id} not found in Redis Hash cache.")
    return None
