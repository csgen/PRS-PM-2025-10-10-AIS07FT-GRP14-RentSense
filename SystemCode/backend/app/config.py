from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    
    # openai
    OPENAI_API_KEY: str

    # cloud database
    CLOUD_DB_HOST: str
    CLOUD_DB_PORT: str
    CLOUD_DB_USERNAME: str
    CLOUD_DB_PASSWORD: str
    CLOUD_DB_DATABASE: str

    # app database
    APP_DB_HOST: str
    APP_DB_PORT: str
    APP_DB_USER: str
    APP_DB_PASSWORD: str
    APP_DB_NAME: str

    # pgAdmin
    PGADMIN_EMAIL: str
    PGADMIN_PASSWORD: str

    # Redis
    REDIS_HOST: str
    REDIS_PORT: str

    class Config:
        env_file = ".env"

settings = Settings()