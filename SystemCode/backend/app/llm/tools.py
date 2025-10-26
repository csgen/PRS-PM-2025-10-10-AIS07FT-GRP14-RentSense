from pydantic import BaseModel, Field
from typing import List, Optional

class EnquiryExtractionTool(BaseModel):
    min_monthly_rent: Optional[int] = Field(
        default=None, 
        description="The minimum monthly rent the user can accept. "
                    "Example 1: For '800 to 1000', set this to 800. "
                    "Example 2: For 'around $1000' OR a single number like 'budget $1000', this implies a +/- 300 range. Calculate and set this to 700 (which is 1000 - 300). "
                    "Example 3: For 'above $800' or 'at least $800', set this to 800."
    )
    
    max_monthly_rent: Optional[int] = Field(
        default=None, 
        description="The maximum monthly rent the user can accept. "
                    "Example 1: For 'under 1000' or 'not more than 1000', set this to 1000. "  # <-- 注意：我们从这里移除了 '$1000'
                    "Example 2: For '800 to 1000', set this to 1000. "
                    "Example 3: For 'around $1000' OR a single number like 'budget $1000', this implies a +/- 300 range. Calculate and set this to 1300 (which is 1000 + 300)."
    )
    
    school_id: Optional[int] = Field(
        default=None, 
        description="The unique identifier of the user's target school.")
    
    target_district_id: Optional[int] = Field(
        default=None, 
        description="The unique identifier of the user's preferred district/area.")
    
    max_school_limit: Optional[int] = Field(
        default=None, 
        description="The maximum commute time (in minutes) the user can accept to reach the school, e.g., 30 in 'within 30 minutes to school'.")
    
    flat_type_preference: Optional[List[str]] = Field(
        default=None, 
        description="List of the user's preferred housing types, e.g., ['Condo', 'HDB'].")
    
    max_mrt_distance: Optional[int] = Field(
        default=None, 
        description="The maximum walking distance (in meters) the user can accept to the nearest MRT station, e.g., 'near MRT' can be interpreted as 500.")

    importance_rent: Optional[int] = Field(
        default=3, ge=1, le=5,
        description="User's importance level for 'rental cost' (1=not important, 3=moderate, 5=very important)."
    )
    importance_location: Optional[int] = Field(
        default=3, ge=1, le=5,
        description="User's importance level for 'location/commute' (1=not important, 3=moderate, 5=very important)."
    )
    importance_facility: Optional[int] = Field(
        default=3, ge=1, le=5,
        description="User's importance level for 'surrounding facilities/safety' (1=not important, 3=moderate, 5=very important)."
    )

    