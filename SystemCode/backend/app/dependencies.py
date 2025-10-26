import openai
from fastapi import Request
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
import redis.asyncio as redis
from typing import AsyncIterator

from app.database.config import engine
from app.database.cache import redis_client


def get_async_openai_client(request: Request) -> openai.AsyncOpenAI:
    return request.app.state.async_openai_client


async_session_maker = async_sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

async def get_async_session() -> AsyncIterator[AsyncSession]:
    async with async_session_maker() as session:
        yield session

async def get_async_redis() -> redis.Redis:
    return redis_client
