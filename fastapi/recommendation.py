from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np
import openai
import json
import re
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from database import RecommendationPlan, RecommendedDay, RecommendationSchedule, RecommendationPlace, Preference, Purpose, PreferencePurpose, SessionLocal, Base
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from typing import List

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic (previously in @app.on_event("startup"))
    global knn_model, data, label_encoders, y_train
    file_path = './final_merged_data.csv'
    data = load_and_preprocess_data(file_path)
    data, label_encoders = encode_and_scale_data(data)
    X_train, X_test, y_train, y_test = split_data(data)
    knn_model = train_knn_model(X_train)
    print("Model trained and ready.")
    
    yield  # Application runs while yielding
    
app = FastAPI(lifespan=lifespan)

# Pydantic model for input validation
class RequestData(BaseModel):
    TRAVEL_PURPOSE: str # purposes
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

# Define the mapping for means of transportation conversion (MVMN_NM)
MVMN_NM_MAP: Dict[str, str] = {
    'CAR': '자가용',
    'PUBLIC_TRANSPORTATION': '대중교통 등'
}
# Define the mapping for TRAVEL_PURPOSE conversion
TRAVEL_PURPOSE_MAP: Dict[str, str] = {
    'SHOPPING': "1;",
    'THEME_PARK': "2;",
    'HISTORIC_SITE': "3;",
    'CITY_TOUR': "4;",
    'OUTDOOR_SPORTS': "5;",
    'CULTURAL_EVENT': "6;",
    'NIGHTLIFE': "7;",
    'CAMPING': "8;",
    'LOCAL_FESTIVAL': "9;",
    'SPA': "10;",
    'EDUCATION': "11;",
    'FILM_LOCATION': "12;",
    'PILGRIMAGE': "13;",
    'WELLNESS': "21;",
    'SNS_SHOT': "22;",
    'HOTEL_STAYCATION': "23;",
    'NEW_TRAVEL_DESTINATION': "24;",
    'PET_FRIENDLY': "25;",
    'INFLUENCER_FOLLOW': "26;",
    'ECO_TRAVEL': "27;",
    'HIKING': "28;"
}

# Define the mapping for style conversion
TRAVEL_STYL_1_MAP: Dict[str, int] = {
    'VERY_NATURE': 1,
    'MODERATE_NATURE': 2,
    'NEUTRAL': 3,
    'MODERATE_CITY': 4,
    'VERY_CITY': 5
}

# Define the mapping for motive conversion
TRAVEL_MOTIVE_1: Dict[str, int] = {
    'ESCAPE': 1,
    'REST': 2,
    'COMPANION_BONDING': 3,
    'SELF_REFLECTION': 4,
    'SOCIAL_MEDIA': 5,
    'EXERCISE': 6,
    'NEW_EXPERIENCE': 7,
    'CULTURAL_EDUCATION': 8,
    'SPECIAL_PURPOSE': 9
}

# Define the mappings
AGE_GRP_MAP: Dict[str, str] = {
    'UNDER_9': "10",
    'TEENS': "10",
    '20S': "20",
    '30S': "30",
    '40S': "40",
    '50S': "50",
    '60S': "60",
    '70_AND_OVER': "70"
}

# Define the mapping for travel status accompany (TRAVEL_STATUS_ACCOMPANY)
TRAVEL_STATUS_ACCOMPANY_MAP: Dict[str, str] = {
    'GROUP_OVER_3': '3인 이상 여행(가족 외)',
    'WITH_CHILD': '자녀 동반 여행',
    'DUO': '2인 여행(가족 외)',
    'SOLO': '나홀로 여행',
    'FAMILY_DUO': '2인 가족 여행',
    'EXTENDED_FAMILY': '3대 동반 여행(친척 포함)'
}

# DTO for returning recommendation plans
class RecommendationPlanDTO(BaseModel):
    recommendation_trip_id: int
    title: str
    description: str

# DTO for the response
class ScheduleItemDTO(BaseModel):
    place: str
    longitude: str
    latitude: str
    address: str

class DayScheduleDTO(BaseModel):
    day: str
    schedule: List[ScheduleItemDTO]

class RecommendationResponseDTO(BaseModel):
    title: str
    description: str
    days: List[DayScheduleDTO]

# Add CORS middleware here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (can restrict to specific origins later)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Load and preprocess the data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, dtype={'TRAVEL_ID': str}, low_memory=False)
    data['MVMN_NM'] = data['MVMN_NM'].ffill()
    return data

# Handle encoding and scaling
def encode_and_scale_data(data):
    label_encoders = {}
    categorical_cols = ['TRAVEL_PURPOSE', 'MVMN_NM', 'AGE_GRP', 'GENDER', 'TRAVEL_STATUS_ACCOMPANY', 'ROAD_ADDR', 'VISIT_AREA_NM']
    
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
    
    return data, label_encoders

# Split the data into train/test sets
def split_data(data):
    X = data.drop(columns=['TRAVEL_ID', 'TRAVELER_ID', 'VISIT_AREA_NM', 'X_COORD', 'Y_COORD'])
    y = pd.get_dummies(data['VISIT_AREA_NM']).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train the KNN model
def train_knn_model(X_train):
    knn_model = NearestNeighbors(n_neighbors=300, algorithm='auto')
    knn_model.fit(X_train)
    return knn_model

# Generate travel recommendations using KNN based on new input data
def get_travel_recommendations(knn_model, new_input_data, data, y_train, label_encoders):
    # Define the expected column order (order must match what the model was trained on)
    expected_columns_order = [
        'TRAVEL_PURPOSE', 'MVMN_NM', 'AGE_GRP', 'GENDER', 
        'TRAVEL_STYL_1', 'TRAVEL_MOTIVE_1', 
        'TRAVEL_STATUS_ACCOMPANY', 'TRAVEL_STATUS_DAYS', 'ROAD_ADDR'
    ]
    
    # Encode categorical input data
    new_input_encoded = {}

    # Handle categorical columns that require encoding
    categorical_cols = ['TRAVEL_PURPOSE', 'MVMN_NM', 'AGE_GRP', 'GENDER', 'TRAVEL_STATUS_ACCOMPANY', 'ROAD_ADDR']
    for col in categorical_cols:
        new_input_encoded[col] = label_encoders[col].transform([new_input_data[col]])[0]

    # Handle integer columns directly without encoding
    int_cols = ['TRAVEL_STYL_1', 'TRAVEL_MOTIVE_1', 'TRAVEL_STATUS_DAYS']
    for col in int_cols:
        new_input_encoded[col] = new_input_data[col]  # Add the integer fields directly

    # Ensure that the encoded input data is ordered correctly according to `expected_columns_order`
    input_array = np.array([new_input_encoded[col] for col in expected_columns_order]).reshape(1, -1)  # Convert to 2D array
    
    # Use the KNN model to find nearest neighbors
    distances, indices = knn_model.kneighbors(input_array)
    
    # Decode the road address back to the original value using LabelEncoder (NO int() conversion)
    test_road_addr_encoded = new_input_encoded['ROAD_ADDR']  # Already encoded as a numeric value
    test_road_addr = label_encoders['ROAD_ADDR'].inverse_transform([test_road_addr_encoded])[0]
    
    # Get the top 20 closest recommendations
    top_20_indices = indices[0]
    top_20_destinations = label_encoders["VISIT_AREA_NM"].inverse_transform(np.argmax(y_train[top_20_indices], axis=1))
    top_20_road_addrs_encoded = data.iloc[top_20_indices]['ROAD_ADDR'].values
    top_20_road_addrs = label_encoders['ROAD_ADDR'].inverse_transform(top_20_road_addrs_encoded)

    # Filter by matching road address
    matching_destinations_and_road_addrs = [
        (destination, road_addr) for destination, road_addr in zip(top_20_destinations, top_20_road_addrs)
        if road_addr == test_road_addr
    ]
    
    # Print matching destinations and road addresses before returning
    print(f"Matching Destinations and Road Addresses: {matching_destinations_and_road_addrs}")
    print(f"Test Road Address: {test_road_addr}")

    return matching_destinations_and_road_addrs, test_road_addr


# Generate travel itinerary using OpenAI API
def create_travel_itinerary(matching_destinations_and_road_addrs, days):
    openai.api_key = ''  # Replace with your OpenAI API key
    
    # Updated prompt to request output in the desired JSON-like structure for multiple days
    prompt_content = f"""
    You are a travel itinerary planner. Based on the user's preferences, create a detailed travel itinerary for {days} days using the provided travel location information. 
    Ensure that each day recommends **different and unique** places to visit from the list of provided locations, with no repetition across days. 
    For each recommended place and restaurant, search for and include the correct address, longitude, and latitude, based on the available data or external knowledge.

    In addition to the schedule, generate one overall **title** and **description** for the entire itinerary in Korean, summarizing the trip as a whole.

    The output should be in the following JSON format, with these specific fields:
    {{
        "title": "<A brief, relevant title for the entire itinerary in Korean>",
        "description": "<A descriptive summary of the entire itinerary in Korean>",
        "days": [
            {{
                "day": "<Only Day Numeric Number>",
                "schedule": [
                    {{
                        "place": "<Exact name of the location>",
                        "longitude": "<longitude of the location (found based on address)>",
                        "latitude": "<latitude of the location (found based on address)>",
                        "fee": "<free or paid (with specific amount in Korean)>",
                        "hours": "<operating hours in Korean>",
                        "address": "<correct location address in Korean, found or validated>"
                    }},
                    {{
                        "restaurant": "<restaurant name>",
                        "longitude": "<longitude of the restaurant (found based on address)>",
                        "latitude": "<latitude of the restaurant (found based on address)>",
                        "address": "<correct restaurant address in Korean, found or validated>"
                    }}
                ]
            }},
            ...
        ]
    }}

    Each day must include 4 different tourist places and 1 unique restaurant, with their respective correct address, longitude, and latitude. 
    If there are not enough provided locations or restaurants in the user preferences, use your knowledge or search for additional places and restaurants to ensure the itinerary is complete. 
    No repetition of locations across days is allowed.

    Do not include non-tourist places like rest areas, gas stations, marts, or convenience stores.

    Create a plan for all {days} days, each with a different set of places and restaurants.
    
    **Extracted travel location information based on user preferences:** 
    {matching_destinations_and_road_addrs}
    """
    
    # GPT 요청 코드 (새로운 인터페이스 사용)
    response = openai.chat.completions.create(
        model="gpt-4",  # engine 대신 model을 사용
        messages=[
            {"role": "system", "content": "You are the person who makes travel plans."},
            {"role": "user", "content": prompt_content}
        ],
        max_tokens=2000
    )

    
    return response

# Save the itinerary to a JSON file in the desired format
def save_itinerary(response):
    response_content = response.choices[0].message.content.strip()

    # Remove non-JSON introductory text like "Here's the 3-day travel plan itinerary..."
    cleaned_content = re.sub(r'^.*?{', '{', response_content, flags=re.DOTALL)
    cleaned_content = re.sub(r'}\s*[^}]*$', '}', cleaned_content)
    cleaned_content = cleaned_content.strip()  # Trim any leading/trailing spaces
    # Print cleaned content to check its validity before parsing
    # print("Cleaned Content: ", cleaned_content)

    wrapped_data = {"travel": cleaned_content}
    # print(f"wrapped_data: {wrapped_data}")
    with open('recommendation_cl.json', 'w', encoding='utf-8') as json_file:
        json.dump(wrapped_data, json_file, ensure_ascii=False, indent=4)

    # Load the JSON file
    # Load the JSON file (assuming it is a string inside the 'travel' key)
    with open('./recommendation_cl.json', 'r', encoding='utf-8') as json_file:
        loaded_data = json.load(json_file)

    # Access the 'travel' value
    travel_string = loaded_data.get("travel", "")

    # Remove 'Day X:' labels and extra markdown artifacts (like ```)# Replace any occurrence of "Day X:" with a comma to properly format JSON
    cleaned_travel_string = re.sub(r'```|Day \d+:', ',', travel_string)

    # Remove unnecessary spaces and ensure proper spacing around commas
    cleaned_travel_string = re.sub(r'\s*,\s*', ',', cleaned_travel_string)

    # Remove extra commas, focusing on replacing ",,," with "," 
    cleaned_travel_string = re.sub(r',,{1,}', ',', cleaned_travel_string)

    # Add brackets to ensure valid JSON array format
    if not cleaned_travel_string.startswith('['):
        cleaned_travel_string = '[' + cleaned_travel_string
    if not cleaned_travel_string.endswith(']'):
        cleaned_travel_string += ']'

    # Remove any trailing comma before closing the array bracket
    cleaned_travel_string = re.sub(r',\s*]', ']', cleaned_travel_string)
    print(f"cleaned_travel_string: {cleaned_travel_string}")
    # Try to load the cleaned string as JSON
    try:
        # Convert cleaned string to a valid JSON object
        travel_data = json.loads(cleaned_travel_string)
        
        # # Print out all the days and their schedules
        # for day in travel_data:
        #     print(f"Day: {day.get('day')}")
        #     for schedule in day.get('schedule', []):
        #         for key, value in schedule.items():
        #             print(f"{key}: {value}")
                    
        # Optionally, you can also save the cleaned JSON back to a file
        with open('cleaned_recommendation_cl.json', 'w', encoding='utf-8') as json_file:
            json.dump(travel_data, json_file, ensure_ascii=False, indent=4)
        print("Cleaned JSON has been saved successfully.")

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    
    return travel_data

@app.get("/recommendations/all", response_model=List[RecommendationPlanDTO])
async def get_all_recommendation(user_id: int = Header(...), db: Session = Depends(get_db)):
    recommendation_plans = db.query(RecommendationPlan).filter_by(user_id=user_id).all()
    
    # Prepare and return the list of recommendations
    return [
        RecommendationPlanDTO(recommendation_trip_id=plan.recommendation_trip_id, title=plan.name, description=plan.description)
        for plan in recommendation_plans
    ]

@app.get("/recommendations/{recommendation_trip_id}", response_model=List[RecommendationResponseDTO])
async def get_recommendation(recommendation_trip_id: str, db: Session = Depends(get_db)):
    # Query the RecommendationPlan by recommendation_trip_id
    recommendation_plan = db.query(RecommendationPlan).filter_by(recommendation_trip_id=recommendation_trip_id).first()

    if not recommendation_plan:
        raise HTTPException(status_code=404, detail="Recommendation plan not found")

    # Fetch associated days and schedules
    recommended_days = db.query(RecommendedDay).filter_by(recommended_trip_id=recommendation_trip_id).all()

    # Prepare the days and schedules in the required format
    days_data = []
    for day in recommended_days:
        # Fetch schedules for the day
        schedules = db.query(RecommendationSchedule).filter_by(day_id=day.day_id).all()

        day_schedules = []
        for schedule in schedules:
            place = schedule.place
            if place:  # This assumes a relationship to RecommendationPlace exists
                # Create a schedule item for each place
                day_schedules.append(ScheduleItemDTO(
                    place=place.place_name,
                    longitude=str(place.longitude),
                    latitude=str(place.latitude),
                    address=place.address
                ))
            elif schedule.restaurant:  # If it's a restaurant, use the restaurant data
                restaurant = schedule.restaurant
                day_schedules.append(ScheduleItemDTO(
                    place=restaurant.place_name,
                    longitude=str(restaurant.longitude),
                    latitude=str(restaurant.latitude),
                    address=restaurant.address
                ))

        # Add the day and its schedules to the response
        days_data.append(DayScheduleDTO(day=str(day.day_number), schedule=day_schedules))

    # Create the response DTO
    response_dto = RecommendationResponseDTO(
        title=recommendation_plan.name,
        description=recommendation_plan.description,
        days=days_data
    )

    return [response_dto]


@app.delete("/recommendations/{recommendation_trip_id}")
async def delete_recommendation(recommendation_trip_id: int, db: Session = Depends(get_db)):
    try:
        # Step 1: Find the RecommendationPlan by ID
        recommendation_plan = db.query(RecommendationPlan).filter_by(recommendation_trip_id=recommendation_trip_id).first()
        
        if not recommendation_plan:
            raise HTTPException(status_code=404, detail="Recommendation plan not found")

        # Step 2: Delete all related RecommendationSchedules
        recommended_days = db.query(RecommendedDay).filter_by(recommended_trip_id=recommendation_trip_id).all()
        
        for day in recommended_days:
            # Get schedules for the day
            schedules = db.query(RecommendationSchedule).filter_by(day_id=day.day_id).all()
            for schedule in schedules:
                # Delete the schedule
                db.delete(schedule)
            
            # Step 3: Delete places associated with the schedules
            places = db.query(RecommendationPlace).filter(RecommendationPlace.place_id.in_(
                [schedule.place_id for schedule in schedules])).all()
            for place in places:
                db.delete(place)
            
            # Delete the day after its schedules and places have been deleted
            db.delete(day)
        
        # Step 4: Delete Preferences and PreferencePurpose related to the recommendation
        preferences = db.query(Preference).filter_by(recommendation_id=recommendation_plan.recommendation_trip_id).all()
        for preference in preferences:
            # Delete all PreferencePurpose records
            preference_purposes = db.query(PreferencePurpose).filter_by(preference_id=preference.preference_id).all()
            for preference_purpose in preference_purposes:
                db.delete(preference_purpose)
            
            # Delete the preference after its related entries are deleted
            db.delete(preference)

        # Step 5: Finally, delete the RecommendationPlan itself
        db.delete(recommendation_plan)

        # Commit all changes
        db.commit()

        return {"message": f"Recommendation with ID {recommendation_trip_id} and related data has been deleted successfully."}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the recommendation: {str(e)}")
    

    
@app.post("/recommendations")
async def recommend(request_data: RequestData, user_id: int = Header(...), db: Session = Depends(get_db)):
    try:
        # Now user_id is available here
        print(f"User ID from header: {user_id}")
        
        # Convert input to dictionary and pass to recommendation function
        input_data = convert_request_to_input(request_data)
        # print(input_data_example)
        new_input_data = input_data.dict()
        print(new_input_data)
        # Generate travel recommendations
        matching_destinations_and_road_addrs, test_road_addr = get_travel_recommendations(
            knn_model, new_input_data, data, y_train, label_encoders
        )

        # If no matching destinations found
        if not matching_destinations_and_road_addrs:
            return {"message": "No matching destinations found."}

        # Generate travel itinerary using OpenAI
        response = create_travel_itinerary(matching_destinations_and_road_addrs, request_data.TRAVEL_STATUS_DAYS)
        
        travel_data = save_itinerary(response)
        # Save the recommendations to the database using user_id
        created_recommendation_id = save_itinerary_to_db(travel_data, user_id, db, request_data)

        return {"message": "Itinerary has been generated and saved.",
                "recommendation_trip_id": created_recommendation_id}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")
    
    except ValueError as e:
        print(e)
    
def save_itinerary_to_db(loaded_data, user_id, db: Session, request_data: RequestData):

    # with open('./cleaned_recommendation_cl.json', 'r', encoding='utf-8') as json_file:
    #     loaded_data = json.load(json_file)
    # Unpack title and description from the loaded_data
    title = loaded_data[0]['title']
    description = loaded_data[0]['description']
    
    print(f"loaded data: {loaded_data}")
    # Create a new RecommendationPlan entry
    recommendation_plan = RecommendationPlan(
        user_id=user_id,  # Use the user_id passed to the function
        name=title,  # Set the title from JSON
        description=description,  # Set the description from JSON
        recommendation_type=request_data.recommendation_type  # Use the appropriate type from request_data
    )
    db.add(recommendation_plan)
    db.commit()
    db.refresh(recommendation_plan)  # Get the new auto-incremented recommendation_trip_id

    # Step 1: Create and save the Preference entry
    preference = Preference(
        recommendation_id=recommendation_plan.recommendation_trip_id,
        style=request_data.TRAVEL_STYL_1,
        motive=request_data.TRAVEL_MOTIVE_1,
        means_of_transportation=request_data.MVMN_NM,
        travel_companion_status=request_data.TRAVEL_STATUS_ACCOMPANY,
        age_group=request_data.AGE_GRP,
        start_date=datetime.strptime(request_data.start_date, '%Y-%m-%d'),
        end_date=datetime.strptime(request_data.end_date, '%Y-%m-%d')
    )
    db.add(preference)
    db.commit()
    db.refresh(preference)

    # Step 2: Create and save the Purpose and PreferencePurpose entries
    purpose = request_data.TRAVEL_PURPOSE  # Assuming TRAVEL_PURPOSE is a list of purposes
    # Check if the purpose already exists, or add it if not
    purpose = Purpose(name=purpose)
    db.add(purpose)
    db.commit()  # Commit immediately after adding a new purpose
    db.refresh(purpose)

    # Check for duplicate (preference_id, purposes_id) before adding to prevent IntegrityError
    existing_preference_purpose = db.query(PreferencePurpose).filter_by(
        preference_id=preference.preference_id,
        purposes_id=purpose.purposes_id
    ).first()

    if not existing_preference_purpose:
        # Create the many-to-many relationship record
        preference_purpose = PreferencePurpose(
            preference_id=preference.preference_id,
            purposes_id=purpose.purposes_id
        )
        db.add(preference_purpose)
    
    db.commit()  # Commit all purposes and preference purposes at once

    # Step 2: Loop through each day in the itinerary using an index
    number_of_days = len(loaded_data[0]['days'])
    print(number_of_days)
    for i in range(number_of_days):
        # Access the day info using index 'i'
        day_info = loaded_data[0]['days'][i]
        print(i)
        print(f"day_info: {day_info}")
        
        day_number = day_info['day']  # Extract day number
        print(day_number)
        current_datetime = datetime.now()  # Use the current date and time
        
        # Create an entry for each day
        recommended_day = RecommendedDay(
            recommended_trip_id=recommendation_plan.recommendation_trip_id,  # Use the auto-incremented ID
            day_number=day_number,
            date=current_datetime  # Use current date and time
        )
        db.add(recommended_day)
        db.flush()  # Flush here to ensure recommended_day.day_id is available
        
        # Loop through each schedule item in the day
        for schedule_order, place in enumerate(day_info['schedule'], start=1):
            if 'place' in place:
                # Handle place entries (add default values if missing)
                latitude = place.get('latitude', 0.0)
                longitude = place.get('longitude', 0.0)
                
                # Create the place entry
                place_entry = RecommendationPlace(
                    place_name=place['place'],
                    address=place['address'],
                    latitude=latitude,
                    longitude=longitude
                )
                db.add(place_entry)
                db.flush()  # Flush to get place_id
                
                # Create a schedule entry for the place
                schedule_entry = RecommendationSchedule(
                    day_id=recommended_day.day_id,
                    place_id=place_entry.place_id,
                    schedule_order=schedule_order
                )
                db.add(schedule_entry)
                db.flush()
            
            elif 'restaurant' in place:
                # Handle restaurant entries
                latitude = place.get('latitude', 0.0)
                longitude = place.get('longitude', 0.0)
                
                # Create the place entry for restaurant
                place_entry = RecommendationPlace(
                    place_name=place['restaurant'],
                    address=place['address'],
                    latitude=latitude,
                    longitude=longitude
                )
                db.add(place_entry)
                db.flush()  # Flush to get place_id
                
                # Create a schedule entry for the restaurant
                schedule_entry = RecommendationSchedule(
                    day_id=recommended_day.day_id,
                    place_id=place_entry.place_id,
                    schedule_order=schedule_order
                )
                db.add(schedule_entry)
                db.flush()
        
        # Commit all days, schedules, and places after processing all entries
        db.commit()

    # Step 4: Save the cleaned JSON to a new file named with recommendation_trip_id
    with open(f'{recommendation_plan.recommendation_trip_id}.json', 'w', encoding='utf-8') as json_file:
        json.dump(loaded_data, json_file, ensure_ascii=False, indent=4)

    # Return the recommendation_trip_id
    return recommendation_plan.recommendation_trip_id



# Function to convert RequestData to InputData
def convert_request_to_input(request_data: RequestData) -> InputData:
    # Convert the TRAVEL_PURPOSE using the defined mapping
    travel_purpose_converted = TRAVEL_PURPOSE_MAP.get(request_data.TRAVEL_PURPOSE)
    if travel_purpose_converted is None:
        raise ValueError(f"Invalid TRAVEL_PURPOSE value: {request_data.TRAVEL_PURPOSE}")
    
    # Convert the style using the defined mapping
    style_converted = TRAVEL_STYL_1_MAP.get(request_data.TRAVEL_STYL_1)
    if style_converted is None:
        raise ValueError(f"Invalid style value: {request_data.TRAVEL_STYL_1}")

    # Convert the motive using the defined mapping
    motive_converted = TRAVEL_MOTIVE_1.get(request_data.TRAVEL_MOTIVE_1)
    if motive_converted is None:
        raise ValueError(f"Invalid TRAVEL_MOTIVE_1 value: {request_data.TRAVEL_MOTIVE_1}")

    # Convert the means of transportation using the defined mapping
    mvmn_nm_converted = MVMN_NM_MAP.get(request_data.MVMN_NM)
    if mvmn_nm_converted is None:
        raise ValueError(f"Invalid MVMN_NM value: {request_data.MVMN_NM}")

    # Convert the travel status accompany using the defined mapping
    travel_status_accompany_converted = TRAVEL_STATUS_ACCOMPANY_MAP.get(request_data.TRAVEL_STATUS_ACCOMPANY)
    if travel_status_accompany_converted is None:
        raise ValueError(f"Invalid TRAVEL_STATUS_ACCOMPANY value: {request_data.TRAVEL_STATUS_ACCOMPANY}")
    
    # Convert the age group using the defined mapping
    age_grp_converted = AGE_GRP_MAP.get(request_data.AGE_GRP)
    if age_grp_converted is None:
        raise ValueError(f"Invalid AGE_GRP value: {request_data.AGE_GRP}")
    
    # Create the InputData object
    input_data = InputData(
        TRAVEL_PURPOSE=travel_purpose_converted,
        MVMN_NM=mvmn_nm_converted,
        AGE_GRP=age_grp_converted,
        GENDER=request_data.GENDER,
        TRAVEL_STYL_1=style_converted,
        TRAVEL_MOTIVE_1=motive_converted,
        TRAVEL_STATUS_ACCOMPANY=travel_status_accompany_converted,
        TRAVEL_STATUS_DAYS=request_data.TRAVEL_STATUS_DAYS,
        ROAD_ADDR=request_data.ROAD_ADDR,
    )
    return input_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
