from fastapi import APIRouter, Depends, status
from fastapi.responses import HTMLResponse
from sqlmodel.ext.asyncio.session import AsyncSession
import openai

from app.dependencies import get_async_session, get_async_openai_client
from app.dependencies import get_async_openai_client
from app.models import EnquiryForm, EnquiryNL, PropertyLocation, RecommendationResponse
from app.handlers import property_handler

from app.dataservice.sql_api.func import query_housing_data_async
from app.dataservice.sql_api.api_model import RequestInfo
import json
from app.dataservice.sql_api.api import fetch_recommend_properties_async
import time


router = APIRouter(prefix="/api/v1/properties", tags=["properties"])


# Submit the questionnaire form
@router.post("/submit-form", response_model=RecommendationResponse, status_code=status.HTTP_201_CREATED)
async def submit_form(
    *,
    db: AsyncSession = Depends(get_async_session),
    client: openai.AsyncOpenAI = Depends(get_async_openai_client),
    enquiry: EnquiryForm
):
    print(f'============enquiry.device_id: {enquiry.device_id}============')
    return await property_handler.submit_form_handler(db=db, client=client, enquiry=enquiry)


# Submit natural language description
@router.post("/submit-description", response_model=RecommendationResponse, status_code=status.HTTP_201_CREATED)
async def submit_description(
    *,
    db: AsyncSession = Depends(get_async_session),
    client: openai.AsyncOpenAI = Depends(get_async_openai_client),
    enquiry: EnquiryNL
):

    return await property_handler.submit_description_handler(db=db, client=client, enquiry=enquiry)


# Get a list of recommended properties
@router.get("/recommendation-no-submit", response_model=RecommendationResponse, status_code=status.HTTP_200_OK)
async def recommendation_no_submit(
    *,
    db: AsyncSession = Depends(get_async_session)
):
    return RecommendationResponse(properties=[])


# Get the map location of a property
@router.post("/map", response_class=HTMLResponse, status_code=status.HTTP_201_CREATED)
async def map(
    *,
    location: PropertyLocation
):
    return await property_handler.map_handler(location=location) 

# 调试函数
def orm_to_dict(obj):
    return {
        c.name: getattr(obj, c.name)
        for c in obj.__table__.columns
        if c.name not in ("geom", "geog")
    }

@router.post("/debug/query-housing-data", response_model=None, status_code=status.HTTP_200_OK)
async def debug_query_housing_data(
    *,
    db: AsyncSession = Depends(get_async_session),
    request_info: RequestInfo
):
    '''
    实例输入：
    {
        "min_monthly_rent": 1000,
        "max_monthly_rent": 2500,
        "school_id": 1,
        "target_district_id": 3
        "max_school_limit": 1500,
        "flat_type_preference": [
            "Condo"
        ],
        "max_mrt_distance": 1500,
        "importance_rent": 3,
        "importance_location": 3,
        "importance_facility": 3
    }
    '''
    start_time = time.time()
    results = await query_housing_data_async(request=request_info)
    data = [orm_to_dict(r) for r in results]

    print(json.dumps(data, indent=2, ensure_ascii=False))

@router.post("/debug/fetch_recommend_properties_async", response_model=None, status_code=status.HTTP_200_OK)
async def debug_query_housing_data(
    *,
    db: AsyncSession = Depends(get_async_session),
    request_info: RequestInfo
):
    results = await fetch_recommend_properties_async(request_info)
    for r in results:
        print(r.model_dump_json())

from app.mockdata.mock import run_mock_user

@router.get("/debug/mock_user_data", response_model=None, status_code=status.HTTP_200_OK)
async def debug_query_housing_data(
    *,
    db: AsyncSession = Depends(get_async_session)
):
    _=await run_mock_user()