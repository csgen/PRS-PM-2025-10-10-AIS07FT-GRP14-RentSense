import os
import redis.asyncio as redis

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
