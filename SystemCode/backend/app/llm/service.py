from typing import List
import json
import asyncio
from fastapi import HTTPException, status
import openai

from app.models import EnquiryForm, EnquiryNL, Property
from .tools import EnquiryExtractionTool
from .prompt import EXTRACTION_PROMPT, EXPLANATION_PROMPT


async def convert_natural_language_to_form(
        *,
        enquiry: EnquiryNL,
        client: openai.AsyncOpenAI
) -> dict:
    
    system_prompt = EXTRACTION_PROMPT

    try:
        completion = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enquiry.requirement_description}
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "EnquiryExtractionTool",
                        "description": "Extract user preferences for rental housing.",
                        "parameters": EnquiryExtractionTool.model_json_schema() 
                    }
                }
            ],
            tool_choice={
                "type": "function", 
                "function": {"name": "EnquiryExtractionTool"}
            }
        )
        
        tool_call = completion.choices[0].message.tool_calls[0]
        if tool_call.function.name == "EnquiryExtractionTool":
            extracted_data = json.loads(tool_call.function.arguments)
            filtered_data = {k: v for k, v in extracted_data.items() if v is not None}
            if enquiry.device_id:
                # name must same as EnquiryForm.device_id
                filtered_data['device_id'] = enquiry.device_id
            return filtered_data
        else:
            return {}

    # API Exception
    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Error communicating with AI model."
            
        )
    
    # Parse Exception
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        print(f"LLM parsing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not understand or parse the requirement description. Please try rephrasing your request."
        )
    

async def _generate_explanation_for_property(
    *,
    enquiry: EnquiryForm,
    prop: Property,
    client: openai.AsyncOpenAI
) -> str:
    user_query_json = enquiry.model_dump_json(indent=2)
    property_data_json = prop.model_dump_json(indent=2)
    
    system_prompt = EXPLANATION_PROMPT.format(
        user_query=user_query_json,
        property_data=property_data_json
    )
    
    try:
        completion = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt}
            ],
            temperature=0.7,
            max_tokens=60
        )
        explanation = completion.choices[0].message.content
        return explanation.strip()

    except openai.OpenAIError as e:
        print(f"OpenAI API error during explanation: {e}")
        fallback_name = prop.name or f"Property ID {prop.property_id}"
        return f"This property ({fallback_name}) is highly recommended based on its strong match to your overall preferences."


async def generate_explanation_for_top_properties(
    *,
    enquiry: EnquiryForm,
    ranked_properties: List[Property],
    client: openai.AsyncOpenAI,
    k: int = 10
) -> List[Property]:
    k = min(k, len(ranked_properties))
    top_k_properties = ranked_properties[:k]

    if not top_k_properties:
        return []
    
    explanation_tasks = []
    for prop in top_k_properties:
        task = _generate_explanation_for_property(
            enquiry=enquiry,
            prop=prop,
            client=client
        )
        explanation_tasks.append(task)

    generate_explanations = await asyncio.gather(*explanation_tasks)

    for i, prop in enumerate(top_k_properties):
        print("generation success!")
        prop.recommand_reason = generate_explanations[i]
        print(prop.model_dump_json(indent=2))

    return top_k_properties
