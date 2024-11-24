from pydantic import BaseModel
from typing import Dict, List
from typing import Optional

# Pydantic model for input validation
class RequestData(BaseModel):
    TRAVEL_PURPOSE: List[str] # purposes
    MVMN_NM: str # preference
    AGE_GRP: str # preference
    GENDER: str
    TRAVEL_STYL_1: str # preferences
    TRAVEL_MOTIVE_1: str # preferences
    TRAVEL_STATUS_ACCOMPANY: str # preference
    TRAVEL_STATUS_DAYS: int
    ROAD_ADDR: str
    recommendation_type: str
    start_date: str # preference
    end_date: str # preference

# Pydantic model for input validation
class InputData(BaseModel):
    TRAVEL_PURPOSE: str # purposes
    MVMN_NM: str # preference
    AGE_GRP: str # preference
    GENDER: str
    TRAVEL_STYL_1: int # preferences
    TRAVEL_MOTIVE_1: int # preferences
    TRAVEL_STATUS_ACCOMPANY: str # preference
    TRAVEL_STATUS_DAYS: int
    ROAD_ADDR: str

class PurposeDTO(BaseModel):
    name: str

class PreferenceDTO(BaseModel):
    preference_id: str
        
# DTO for returning recommendation plans
class RecommendationPlanDTO(BaseModel):
    recommendation_trip_id: int
    title: str
    description: str

# DTO for the response
class RecommendationPlaceDTO(BaseModel):
    place: str
    longitude: Optional[str] = None
    latitude: Optional[str] = None
    address: Optional[str] = None

class RecommendationScheduleDTO(BaseModel):
    order: str
    place: RecommendationPlaceDTO

class DayScheduleDTO(BaseModel):
    dayNumber: str
    date: str
    schedule: List[RecommendationScheduleDTO]

class RecommendationResponseDTO(BaseModel):
    title: str
    description: str
    recommendation_type: str
    start_date: str
    end_date: str
    days: List[DayScheduleDTO]
    preference_id: Optional[str] = None

class PreferenceResponseDTO(BaseModel):
    recommendation_trip_id: int
    user_id: int
    travel_status_days: int
    style: int
    motive: int
    means_of_transportation: str
    travel_companion_status: str
    age_group: str
    purposes: str
    gender: str