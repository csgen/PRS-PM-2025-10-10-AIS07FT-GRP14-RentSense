import os
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import create_async_engine
from dotenv import load_dotenv

load_dotenv()

ASYNC_DATABASE_URL = os.getenv("APP_DATABASE_URL")

engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=True,
    future=True
)

async def create_db_and_tables():
    async with engine.begin() as conn:
        # await conn.run_sync(SQLModel.metadata.drop_all) # if reset needed
        await conn.run_sync(SQLModel.metadata.create_all)
