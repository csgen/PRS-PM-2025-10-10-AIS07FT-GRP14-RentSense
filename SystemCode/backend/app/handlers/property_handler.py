from typing import List
import openai
from fastapi import status, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models import EnquiryForm, EnquiryNL, PropertyLocation, Property, RecommendationResponse
from app.database.crud import enquiry_crud, recommendation_crud
from app.database import cache
from app.services import recommendation_service as rec_service
from app.services import map_service as map_service
from app.llm import service as llm_service


async def submit_form_handler(
    *,
    db: AsyncSession,
    client: openai.AsyncOpenAI,
    enquiry: EnquiryForm
) -> RecommendationResponse:

    # save enquiry to db and cache
    enquiry_entity = await enquiry_crud.save_enquiry(db=db, enquiry=enquiry)

    # get TopN recommendation
    properties = await rec_service.fetch_recommend_properties(enquiry)

    # multi-objective optimization ranking
    ranked_properties: List[Property] = await rec_service.multi_objective_optimization_ranking(
        db=db,
        enquiry=enquiry, 
        propertyList=properties
    )

    # LLM generate natural language reason for recommendation
    top_k_with_explanations = await llm_service.generate_explanation_for_top_properties(
        enquiry=enquiry,
        ranked_properties=ranked_properties,
        client=client,
        k = 2 # Top10
    )

    # save recommendation result to db and cache
    await recommendation_crud.save_recommendation(
        eid=enquiry_entity.eid if enquiry_entity else None, 
        db=db, 
        properties=top_k_with_explanations
    )

    await cache.save_latest_properties_as_hash(properties=top_k_with_explanations)

    return RecommendationResponse(properties=top_k_with_explanations)


async def submit_description_handler(
    *,
    db: AsyncSession,
    client: openai.AsyncOpenAI,
    enquiry: EnquiryNL,
) -> RecommendationResponse:

    # natural language -> dict
    extracted_dict = await llm_service.convert_natural_language_to_form(enquiry=enquiry, client=client)

    # check required fields
    missing_fields = _getMissingField(extracted_dict)
    if missing_fields:
        print(f'missing_fields: {missing_fields}')
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "code": 42201,
                "message": f"Missing necessary information: {', '.join(missing_fields)}.",
                "missing_fields": missing_fields
            }
        )
    
    # create form
    try:
        enquiry_form = EnquiryForm.model_validate(extracted_dict)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error validating extracted preferences from LLM: {e}"
        )

    print(f'LLM success! \
          \n EnquiryNL: {enquiry.model_dump_json(indent=2)}\
          \n EnquiryForm: {enquiry_form.model_dump_json(indent=2)}')
    
    return await submit_form_handler(db=db, client=client, enquiry=enquiry_form)


def _getMissingField(extracted_dict: dict) -> list:
    required_fields = ['min_monthly_rent', 'max_monthly_rent', 'school_id']
    return [
        field for field in required_fields 
        if (field not in extracted_dict) or (extracted_dict.get(field) is None)
    ]

async def map_handler(
    *,
    location: PropertyLocation   
) -> HTMLResponse:
    
    #fetch map html page from folium
    html_content = await map_service.fetch_map_page(location=location)
    return HTMLResponse(content=html_content)
