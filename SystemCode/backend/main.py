import openai
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database.config import create_db_and_tables
from app.routes.property import router as property_router
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await create_db_and_tables()
    app.state.async_openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    yield

    # Shutdown
    await app.state.async_openai_client.close()

app = FastAPI(
    title="IRRS API",
    description="API for IntelligentRentingREcommendatinoSystem",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(property_router)

@app.get("/")
async def root():
    return {"message": "Welcome to IRRS"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
