from fastapi import APIRouter, Depends, status
from fastapi.responses import HTMLResponse
from sqlmodel.ext.asyncio.session import AsyncSession
import openai

from app.dependencies import get_async_session, get_async_openai_client
from app.dependencies import get_async_openai_client
from app.models import EnquiryForm, EnquiryNL, PropertyLocation, RecommendationResponse
from app.handlers import property_handler


router = APIRouter(prefix="/api/v1/properties", tags=["properties"])


# Submit the questionnaire form
@router.post("/submit-form", response_model=RecommendationResponse, status_code=status.HTTP_201_CREATED)
async def submit_form(
    *,
    db: AsyncSession = Depends(get_async_session),
    client: openai.AsyncOpenAI = Depends(get_async_openai_client),
    enquiry: EnquiryForm
):

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
