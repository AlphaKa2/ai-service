from datetime import datetime
import json
import logging
import time
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import RedirectResponse
import easyocr_url
import whisper_url
import youtube_info
import youtube_recommendation
import torch
import pandas as pd
import numpy as np
import re
import openai
import boto3
import os
import subprocess
import pickle
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler
from elasticsearch import Elasticsearch, helpers
from fastapi import BackgroundTasks
from datetime import timedelta
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from contextlib import asynccontextmanager
from typing import Dict, List
from typing import Optional
from database import (
    RecommendationPlan,
    RecommendationDay,
    RecommendationSchedule,
    RecommendationPlace,
    Preference,
    Purpose,
    PreferencePurpose,
    SessionLocal,
    Base,
    YoutubeVideo
)
from mapping import (
    MVMN_NM_MAP,
    TRAVEL_PURPOSE_MAP,
    GENDER_MAP,
    GENDER_REVERSE_MAP,
    TRAVEL_STYL_1_MAP,
    TRAVEL_MOTIVE_1,
    AGE_GRP_MAP,
    TRAVEL_STATUS_ACCOMPANY_MAP,
    TRANSPORTATION_MAP
)
from dto import (
    RequestData,
    InputData,
    RecommendationPlanDTO,
    RecommendationPlaceDTO,
    RecommendationScheduleDTO,
    DayScheduleDTO,
    RecommendationResponseDTO,
    PreferenceResponseDTO
)


@asynccontextmanager
async def lifespan(app: FastAPI):    
    """Startup tasks to initialize encoders and scalers."""
    logger.info("Server is starting. Checking for necessary files...")
    global encoder, scaler, mlb  # Access global variables
    try:
        # Load encoders at startup
        encoders = load_encoders()
        encoder = encoders["encoder"]
        scaler = encoders["scaler"]
        mlb = encoders["mlb"]
        logger.info("Encoders and scalers are ready for use.")
        # 사용 전 확인
        if not isinstance(encoder, OneHotEncoder):
            raise TypeError("Loaded encoder is not a valid OneHotEncoder instance.")
        if not isinstance(scaler, StandardScaler):
            raise TypeError("Loaded scaler is not a valid StandardScaler instance.")
        if not isinstance(mlb, MultiLabelBinarizer):
            raise TypeError("Loaded mlb is not a valid MultiLabelBinarizer instance.")

    except Exception as e:
        logger.error(f"Failed to initialize encoders on startup: {str(e)}")
        raise RuntimeError(f"Startup initialization failed: {str(e)}")
    
    yield  # Application runs while yielding
    
app = FastAPI(lifespan=lifespan)

# S3 자격증명
session = boto3.Session(
    aws_access_key_id='access_key',
    aws_secret_access_key='access_key',
    region_name='name'
)

# S3 및 MySQL 연결 설정
s3 = session.client('s3')
bucket_name = 'name'
directory_name = 'name'

# Elasticsearch 연결
es_host = "host"
es_user = "user"
es_password = "password"
es_index = 'index'
es = Elasticsearch(
    hosts=[es_host],
    basic_auth=(es_user, es_password)
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock UserProfile class and service
class UserProfile:
    def __init__(self, user_id: str, username: str):
        self.user_id = user_id
        self.username = username

# Example service to get UserProfile from headers
def get_user_profile_from_header(request: Request) -> UserProfile:
    user_id = request.headers.get("X-User-Id")
    username = request.headers.get("X-Username")
    
    if not user_id or not username:
        raise HTTPException(status_code=400, detail="User information missing in headers")
    
    return UserProfile(user_id=user_id, username=username)

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

# Data encoding and vectorization using preloaded encoders
def encode_and_vectorize_data(data):
    try:
        logger.info("Start vectorized")
        # Define columns
        categorical_cols = ['TRAVEL_PURPOSE', 'MVMN_NM', 'GENDER', 'TRAVEL_STYL_1',
                            'TRAVEL_STATUS_ACCOMPANY', 'ROAD_ADDR']
        numerical_cols = ['AGE_GRP', 'TRAVEL_STATUS_DAYS', 'TRAVEL_MOTIVE_1']

        # Multi-label encoding for 'TRAVEL_PURPOSE'
        data['TRAVEL_PURPOSE'] = data['TRAVEL_PURPOSE'].apply(
            lambda x: [item for item in x.split(';') if item] if pd.notna(x) else []
        )
        travel_purpose_encoded = mlb.transform(data['TRAVEL_PURPOSE'])
        travel_purpose_df = pd.DataFrame(travel_purpose_encoded, columns=[f"TRAVEL_PURPOSE_{cls}" for cls in mlb.classes_])

        # Scaling numerical fields
        numerical_data = scaler.transform(data[numerical_cols].astype(float))
        numerical_df = pd.DataFrame(numerical_data, columns=numerical_cols)
        
        # encoder 객체 로드 후 업데이트
        if hasattr(encoder, 'sparse'):
            encoder.sparse_output = encoder.sparse
            del encoder.sparse  # 오래된 속성 제거

        # One-Hot Encoding for other categorical fields
        other_categorical = data[categorical_cols].drop('TRAVEL_PURPOSE', axis=1).astype(str)
        encoded_categorical = encoder.transform(other_categorical)
        encoded_feature_names = encoder.get_feature_names_out(other_categorical.columns)
        encoded_df = pd.DataFrame(encoded_categorical, columns=encoded_feature_names)

        # Combine all features
        final_vector = pd.concat([numerical_df.reset_index(drop=True),
                                   travel_purpose_df.reset_index(drop=True),
                                   encoded_df.reset_index(drop=True)], axis=1)

        # Log vector dimensions and sample
        logger.info(f"Final vector shape: {final_vector.shape}")
        logger.debug(f"Sample vector: {final_vector.iloc[0].tolist()}")

        return final_vector
    except Exception as e:
        logger.error(f"Error during encoding and vectorization: {str(e)}")
        raise ValueError(f"Failed to encode and vectorize data: {str(e)}")


def get_travel_recommendations(new_input_data):
    try:
        # Step 1: Encode and vectorize input data
        input_df = pd.DataFrame([new_input_data])  # Create a DataFrame with one row
        logger.info(input_df["TRAVEL_PURPOSE"])
        print(input_df["TRAVEL_PURPOSE"])
        logger.info("CONVERT")

        # Step 2: Ensure multi-label field TRAVEL_PURPOSE is formatted as a semicolon-delimited string
        if isinstance(input_df.at[0, "TRAVEL_PURPOSE"], list):
            input_df["TRAVEL_PURPOSE"] = input_df["TRAVEL_PURPOSE"].apply(lambda x: ";".join(x))
        elif isinstance(input_df.at[0, "TRAVEL_PURPOSE"], str):
            pass  # Already in string format, nothing to do
        else:
            raise ValueError("TRAVEL_PURPOSE must be a list or a semicolon-delimited string.")

        logger.info(input_df["TRAVEL_PURPOSE"])

        # Encode and vectorize data
        encoded_vector = encode_and_vectorize_data(input_df)
        logger.info("Finished vectorized")

        # Convert DataFrame to numpy array and then to list
        query_vector = encoded_vector.to_numpy().tolist()[0]

        # Step 3: Dimensionality Check
        expected_dims = 239  # Replace with your expected dimensions
        if len(query_vector) != expected_dims:
            logger.error(f"[ERROR] Vector dimensionality mismatch: Expected {expected_dims}, but got {len(query_vector)}")
            raise ValueError(f"Vector dimensionality mismatch: Expected {expected_dims}, but got {len(query_vector)}")
        
        logger.info("dim success")
        # Step 4: Type Check (Ensure all values are float)
        if not all(isinstance(value, (float, int)) for value in query_vector):
            logger.error("[ERROR] Vector contains non-float values")
            raise ValueError("Vector contains non-float values")
        
        logger.info("float success")
        road_address_value = input_df["ROAD_ADDR"].iloc[0]  # 첫 번째 행의 ROAD_ADDR 값 추출 (단일 값)
        logger.info(f"road_address_value: {road_address_value}")
        # Elasticsearch query for the 10 closest vectors
        es_query = {
            "size": 10,  # 코사인 유사도가 높은 최대 10개 문서 반환
            "_source": ["visit_area", "road_address"],  # 반환 필드 지정
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": [
                                {
                                    "match": {
                                        "road_address": {
                                            "query": road_address_value,
                                            "operator": "and"  # Enforces all terms must match
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.queryVector, 'features_vector') + 1.0",  # 코사인 유사도 계산
                        "params": {
                            "queryVector": query_vector  # 쿼리 벡터
                        }
                    }
                }
            }
        }
        
        mapping = es.indices.get_mapping(index=es_index)
        print(f"mapping: {mapping}")


        response = es.search(index=es_index, body=es_query)

        # 결과를 저장할 리스트
        matching_destinations_and_road_addrs = []

        # 결과 출력
        for doc in response["hits"]["hits"]:
            print(f"Document ID: {doc['_id']}")
            print(f"Visit Area: {doc['_source']['visit_area']}")
            print(f"Road Address: {doc['_source']['road_address']}")
            
            destination = doc["_source"].get("visit_area", "Unknown")  # visit_area 필드 가져오기
            road_addr = doc["_source"].get("road_address", "Unknown")  # road_address 필드 가져오기
    
            # 튜플 형식으로 저장
            matching_destinations_and_road_addrs.append((destination, road_addr))

        return matching_destinations_and_road_addrs

    except Exception as e:
        logger.error(f"Error fetching travel recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching travel recommendations: {e}")





# Generate travel itinerary using OpenAI API
def create_travel_itinerary(matching_destinations_and_road_addrs, days):
    openai.api_key = 'api_key'  # Replace with your OpenAI API key
    
    # Updated prompt with strict formatting instructions for generating a travel itinerary in JSON format
    prompt_content = f"""
    You are a travel itinerary planner. Based on the user's preferences, create a detailed travel itinerary for {days} days using the provided travel location information.

    **Important Formatting Instructions**:
    1. Ensure that the output strictly matches the specified JSON structure without any deviation.
    2. Use **Korean** for the title, description, and addresses, as well as exact information for longitude, latitude, and place names.
    3. Avoid using English or placeholders like '<>' in the output; ensure all fields contain the correct information based on available data.

    The output must follow this exact JSON format:
    {{
        "title": "<A brief, relevant title for the entire itinerary in Korean>",
        "description": "<A descriptive summary of the entire itinerary in Korean>",
        "days": [
            {{
                "day": "<Numeric day number only>",
                "schedule": [
                    {{
                        "place": "<Exact name of the location>",
                        "longitude": "<Longitude of the location, accurately found based on the address>",
                        "latitude": "<Latitude of the location, accurately found based on the address>",
                        "address": "<Correct location address in Korean>"
                    }},
                    {{
                        "restaurant": "<Exact name of the restaurant>",
                        "longitude": "<Longitude of the restaurant, accurately found based on the address>",
                        "latitude": "<Latitude of the restaurant, accurately found based on the address>",
                        "address": "<Correct restaurant address in Korean>"
                    }}
                ]
            }},
            ...
        ]
    }}

    **Content Requirements**:
    1. Each day must include exactly **4 unique tourist places** and **1 unique restaurant**, with no repetition across days. If a tourist place has already been included on a previous day, exclude it and replace it with a different one to ensure variety.
    2. Each place and restaurant must contain the correct address, longitude, and latitude found from available data or external knowledge.
    3. If there are not enough provided locations or restaurants in the user preferences, generate appropriate places or restaurants based on the destination. Ensure that these places or restaurants actually exist at the destination.
    4. Exclude non-tourist places such as rest areas, gas stations, marts, convenience stores, and subway stations, or any other locations that are not strictly considered tourist attractions.

    **Extracted travel location information based on user preferences**:
    {matching_destinations_and_road_addrs}

    Please generate an itinerary for each of the {days} days, ensuring all formatting and content guidelines are followed exactly.
    """


    
    # GPT 요청 코드 (새로운 인터페이스 사용)
    response = openai.chat.completions.create(
        model="gpt-4",  # engine 대신 model을 사용
        messages=[
            {"role": "system", "content": "You are the person who makes travel plans."},
            {"role": "user", "content": prompt_content}
        ],
        max_tokens=3000,
        temperature=0.7
    )

    
    return response

# Save the itinerary to a JSON file in the desired format
def save_itinerary(response):
    response_content = response.choices[0].message.content.strip()

    # Remove non-JSON introductory text like "Here's the 3-day travel plan itinerary..."
    cleaned_content = re.sub(r'^.*?{', '{', response_content, flags=re.DOTALL)
    cleaned_content = re.sub(r'}\s*[^}]*$', '}', cleaned_content)
    cleaned_content = cleaned_content.strip()  # Trim any leading/trailing spaces

    wrapped_data = {"travel": cleaned_content}
    # print(f"wrapped_data: {wrapped_data}")
    with open('recommendation_cl.json', 'w', encoding='utf-8') as json_file:
        json.dump(wrapped_data, json_file, ensure_ascii=False, indent=4)

    # Load the JSON file
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
                    
        # Optionally, you can also save the cleaned JSON back to a file
        with open('cleaned_recommendation_cl.json', 'w', encoding='utf-8') as json_file:
            json.dump(travel_data, json_file, ensure_ascii=False, indent=4)
        print("Cleaned JSON has been saved successfully.")

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    
    return travel_data

@app.get("/recommendations/all", response_model=List[RecommendationPlanDTO])
async def get_all_recommendation(user_id: str = Header(..., alias="X-User-Id"),
    user_role: str = Header(..., alias="X-User-Role"),
    user_profile: str = Header(..., alias="X-User-Profile"),
    user_nickname: str = Header(..., alias="X-User-Nickname"), db: Session = Depends(get_db)):
    recommendation_plans = db.query(RecommendationPlan).filter_by(user_id=user_id).all()
    
    # Prepare and return the list of recommendations
    return [
        RecommendationPlanDTO(recommendation_trip_id=plan.recommendation_trip_id, title=plan.name, description=plan.description)
        for plan in recommendation_plans
    ]

@app.get("/recommendations/{recommendation_trip_id}", response_model=RecommendationResponseDTO)
async def get_recommendation(recommendation_trip_id: str,
                             user_id: str = Header(..., alias="X-User-Id"),
                             db: Session = Depends(get_db)):
    # Query the RecommendationPlan by recommendation_trip_id
    recommendation_plan = db.query(RecommendationPlan).filter_by(recommendation_trip_id=recommendation_trip_id).first()
    if not recommendation_plan:
        raise HTTPException(status_code=404, detail="존재하지 않는 여행입니다.")
    
    # Check if the user_id matches
    if str(recommendation_plan.user_id) != user_id:
        raise HTTPException(status_code=403, detail="해당 여행에 대한 권한이 없습니다.")
    
    # Fetch associated days and schedules
    recommendation_days = db.query(RecommendationDay).filter_by(recommendation_trip_id=recommendation_trip_id).all()
    days_data = []
    for day in recommendation_days:
        # Fetch schedules for the day
        schedules = db.query(RecommendationSchedule).filter_by(day_id=day.day_id).all()
        day_schedules = [
            RecommendationScheduleDTO(
                order=str(schedule.schedule_order),
                place=RecommendationPlaceDTO(
                    place=schedule.place.place_name,
                    longitude=str(schedule.place.longitude) if schedule.place.longitude else None,
                    latitude=str(schedule.place.latitude) if schedule.place.latitude else None,
                    address=schedule.place.address
                )
            )
            for schedule in schedules if schedule.place
        ]
        days_data.append(DayScheduleDTO(
            dayNumber=str(day.day_number),
            date=day.date.isoformat(),
            schedule=day_schedules
        ))

    # Initialize `preference_id` as None
    preference_id = None

    # Only fetch preference if recommendation_type is AI-GENERATED
    if recommendation_plan.recommendation_type == "AI-GENERATED":
        preference = db.query(Preference).filter_by(recommendation_id=recommendation_plan.recommendation_trip_id).first()
        if preference:
            preference_id = str(preference.preference_id)  # Convert to string for consistency

    # Construct the response DTO using Pydantic model instances
    response_dto = RecommendationResponseDTO(
        title=recommendation_plan.name,
        description=recommendation_plan.description,
        recommendation_type=recommendation_plan.recommendation_type,
        start_date=recommendation_plan.start_date.isoformat() if recommendation_plan.start_date else None,
        end_date=recommendation_plan.end_date.isoformat() if recommendation_plan.end_date else None,
        days=days_data,
        preference_id=preference_id
    )
    
    return response_dto



@app.delete("/recommendations/{recommendation_trip_id}")
async def delete_recommendation(recommendation_trip_id: int,
                                user_id: str = Header(..., alias="X-User-Id"),
                                db: Session = Depends(get_db)):
    try:
        # Step 1: Find the RecommendationPlan by ID
        recommendation_plan = db.query(RecommendationPlan).get(recommendation_trip_id)
        if not recommendation_plan:
            raise HTTPException(status_code=404, detail="존재하지 않는 여행입니다.")
        
        if str(recommendation_plan.user_id) != user_id:
            raise HTTPException(status_code=403, detail="해당 여행에 대한 권한이 없습니다.")
    
        # Step 2: Delete the RecommendationPlan (cascade will handle related entities)
        db.delete(recommendation_plan)
        db.commit()

        return {"data": recommendation_trip_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the recommendation: {str(e)}")

    
@app.post("/recommendations")
async def recommend(
    request_data: RequestData,
    user_id: str = Header(..., alias="X-User-Id"),
    user_role: str = Header(..., alias="X-User-Role"),
    user_profile: str = Header(..., alias="X-User-Profile"),
    user_nickname: str = Header(..., alias="X-User-Nickname"),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        # Log the received user ID
        print(f"User ID from header: {user_id}")
        
        # Convert input to dictionary and pass to recommendation function
        input_data = convert_request_to_input(request_data)
        new_input_data = input_data.dict()
        print(new_input_data)
        
        # Schedule the task to run in the background
        background_tasks.add_task(process_recommendation_task, new_input_data, user_id, db, request_data)
        
        # Respond immediately with a 200 OK status
        return {"message": "요청이 성공적으로 처리되었습니다."}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")

def process_recommendation_task(new_input_data, user_id, db, request_data):
    try:
        # Generate travel recommendations
        matching_destinations_and_road_addrs= get_travel_recommendations(new_input_data)
        
        # Check if any matching destinations were found
        if not matching_destinations_and_road_addrs:
            print("No matching destinations found.")
            return
        
        print(matching_destinations_and_road_addrs)
        # Generate travel itinerary using OpenAI
        response = create_travel_itinerary(matching_destinations_and_road_addrs, request_data.TRAVEL_STATUS_DAYS)
        travel_data = save_itinerary(response)
        
        # Save the recommendations to the database using user_id
        created_recommendation_id = save_itinerary_to_db(travel_data, user_id, db, request_data)
        
        print("Itinerary generation and saving completed.")
        return created_recommendation_id
    
    except Exception as e:
        print(f"An error occurred in the background task: {str(e)}")

@app.get("/preferences/{preference_id}", response_model=PreferenceResponseDTO)
async def get_preference(preference_id: str, db: Session = Depends(get_db)):
    try:
        # Fetch the Preference entry by ID
        preference = db.query(Preference).filter_by(preference_id=preference_id).first()
        if not preference:
            raise HTTPException(status_code=404, detail="존재하지 않는 성향정보입니다.")

        # Fetch the associated Recommendation entry to get the user_id
        recommendation = db.query(RecommendationPlan).filter_by(recommendation_trip_id=preference.recommendation_id).first()
        if not recommendation:
            raise HTTPException(status_code=404, detail="존재하지 않는 여행입니다.")

        # Fetch the associated purposes for this preference
        preference_purposes = db.query(PreferencePurpose).filter_by(preference_id=preference_id).all()
        purposes_ids = []
        for preference_purpose in preference_purposes:
            purposes_ids.append(str(preference_purpose.purposes_id))  # Collect purpose IDs as strings

        # Join the purpose IDs with ";" to form the required string
        purposes_str = ";".join(purposes_ids) + ";" if purposes_ids else ""

        # Build the response DTO
        response_dto = PreferenceResponseDTO(
            recommendation_trip_id=preference.recommendation_id,
            user_id=recommendation.user_id,  # Use user_id from the Recommendation table
            travel_status_days=preference.travel_status_days,
            style=TRAVEL_STYL_1_MAP.get(preference.style),
            motive=TRAVEL_MOTIVE_1.get(preference.motive),
            means_of_transportation=TRANSPORTATION_MAP.get(preference.means_of_transportation),
            travel_companion_status=TRAVEL_STATUS_ACCOMPANY_MAP.get(preference.travel_companion_status),
            age_group=AGE_GRP_MAP.get(preference.age_group),
            purposes=purposes_str,  # Pass the formatted string
            gender=GENDER_REVERSE_MAP.get(preference.gender)
        )
        return response_dto

    except Exception as e:
        logger.error(f"Error fetching preference: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# 쿠키 파일 경로 설정
COOKIES_FILE_PATH = "./cookies.txt"

# yt-dlp 다운로드 함수
def download_video_with_cookies(url: str):
    command = [
        "yt-dlp",
        "--cookies", COOKIES_FILE_PATH,  # 쿠키 파일 사용
        url
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail="비디오 다운로드 실패")


@app.post("/process-url")
async def process_url(url: str, user_id: int):
    db = SessionLocal()
    try:
        existing_video = db.query(YoutubeVideo).filter_by(url=url).first()
        if existing_video:
            return JSONResponse(content={"travel_id": existing_video.recommendation_trip_id, "route": existing_video})

        # yt-dlp 다운로드 함수 호출
        download_video_with_cookies(url)

        # YouTube 비디오 정보와 OCR 텍스트 추출
        youtube_info_data = youtube_info.extract_video_info(url)  # youtube_info에서 정보 추출
        easyocr_text = easyocr_url.easy_ocr_function(url)  # easyocr_url에서 OCR 텍스트 추출
        whisper_text = whisper_url.get_youtube_transcript(url)  # whisper_url에서 유튜브 텍스트 추출

        # Whisper 텍스트가 리스트인 경우 결합
        if isinstance(whisper_text, list):
            whisper_text = "\n".join(whisper_text)

        # 여행 추천 경로 생성
        travel_route_json = youtube_recommendation.generate_travel_route(
            easyocr_text,
            whisper_text,
            url,
            user_id,
            db,
            "access_key"
        )

        return JSONResponse(content={"message": "Travel route and associated data saved successfully."})
        
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Database integrity error: {str(e.orig)}")
    except ValueError as ve:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Value error: {str(ve)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Unknown error: {str(e)}")
    
    finally:
        db.close()



def save_itinerary_to_db(loaded_data, user_id, db: Session, request_data: RequestData):
    try:
        print("start_db")
        # Unpack title and description from the loaded_data
        title = loaded_data[0]['title']
        description = loaded_data[0]['description']
        
        # Create a new RecommendationPlan entry
        recommendation_plan = RecommendationPlan(
            user_id=user_id,  # Use the user_id passed to the function
            name=title,  # Set the title from JSON
            description=description,  # Set the description from JSON
            recommendation_type=request_data.recommendation_type,  # Use the appropriate type from request_data
            start_date=datetime.strptime(request_data.start_date, '%Y-%m-%d'),
            end_date=datetime.strptime(request_data.end_date, '%Y-%m-%d')
        )
        db.add(recommendation_plan)
        db.commit()
        db.refresh(recommendation_plan)  # Get the new auto-incremented recommendation_trip_id
        
#         print("complete plan")
        # Step 1: Create and save the Preference entry
        preference = Preference(
            recommendation_id=recommendation_plan.recommendation_trip_id,
            style=request_data.TRAVEL_STYL_1,
            motive=request_data.TRAVEL_MOTIVE_1,
            means_of_transportation=request_data.MVMN_NM,
            travel_companion_status=request_data.TRAVEL_STATUS_ACCOMPANY,
            age_group=request_data.AGE_GRP,
            road_addr=request_data.ROAD_ADDR,
            travel_status_days=request_data.TRAVEL_STATUS_DAYS,
            gender=GENDER_MAP.get(request_data.GENDER)
        )
        db.add(preference)
        db.commit()
        db.refresh(preference)

        # Step 2: Create and save the Purpose and PreferencePurpose entries
        purposes = request_data.TRAVEL_PURPOSE  # Assuming TRAVEL_PURPOSE is a list of purposes

        for purpose_name in purposes:
            # Retrieve the purpose entry from the database
            purpose = db.query(Purpose).filter_by(name=purpose_name).first()
            
            if not purpose:
                raise ValueError(f"Purpose '{purpose_name}' not found in database.")
            
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

        # Commit all purposes and preference purposes at once
        db.commit()

        
#         print("complete preference_purpose")
        
        # Step 2: Loop through each day in the itinerary using an index
        number_of_days = len(loaded_data[0]['days'])
        start_date = datetime.strptime(request_data.start_date, '%Y-%m-%d')  # Parse the start date

        for i in range(number_of_days):
            day_info = loaded_data[0]['days'][i]
            day_number = day_info['day']  # Extract day number

            # Increment the date by i days to ensure each day is unique
            current_date = start_date + timedelta(days=i)

            # Create an entry for each day
            recommendation_day = RecommendationDay(
                recommendation_trip_id=recommendation_plan.recommendation_trip_id,  # Link the trip_id
                day_number=day_number,
                date=current_date  # Set the incremented date
            )
            db.add(recommendation_day)
            db.flush()  # Ensure recommended_day.day_id is available
#             print("complete recommendation_day")
            # Loop through each schedule item in the day
            for schedule_order, place in enumerate(day_info['schedule'], start=1):
                if 'place' in place:
                    # Handle place entries
                    latitude = place.get('latitude', 0.0)
                    longitude = place.get('longitude', 0.0)

                    # Create or check for an existing place entry (to avoid duplicates)
                    place_entry = db.query(RecommendationPlace).filter_by(
                        place_name=place['place'],
                        address=place['address'],
                        latitude=latitude,
                        longitude=longitude
                    ).first()

                    if not place_entry:
                        # Create a new place entry if not found
                        place_entry = RecommendationPlace(
                            place_name=place['place'],
                            address=place['address'],
                            latitude=latitude,
                            longitude=longitude
                        )
                        db.add(place_entry)
                        db.flush()  # Flush to get place_id
#                         print("complete recommendation_place")
                    # Create a schedule entry for the place
                    schedule_entry = RecommendationSchedule(
                        day_id=recommendation_day.day_id,
                        place_id=place_entry.place_id,
                        schedule_order=schedule_order,
                        created_at=datetime.now()  # Add created_at field
                    )
                    db.add(schedule_entry)
#                     print("complete recommendation_schedule")

                elif 'restaurant' in place:
                    # Handle restaurant entries similarly
                    latitude = place.get('latitude', 0.0)
                    longitude = place.get('longitude', 0.0)

                    # Create or check for an existing restaurant entry (to avoid duplicates)
                    place_entry = db.query(RecommendationPlace).filter_by(
                        place_name=place['restaurant'],
                        address=place['address'],
                        latitude=latitude,
                        longitude=longitude
                    ).first()

                    if not place_entry:
                        # Create a new restaurant entry if not found
                        place_entry = RecommendationPlace(
                            place_name=place['restaurant'],
                            address=place['address'],
                            latitude=latitude,
                            longitude=longitude
                        )
                        db.add(place_entry)
                        db.flush()  # Flush to get place_id
#                         print("complete recommendation_place")

                    # Create a schedule entry for the restaurant
                    schedule_entry = RecommendationSchedule(
                        day_id=recommendation_day.day_id,
                        place_id=place_entry.place_id,
                        schedule_order=schedule_order,
                        created_at=datetime.now()  # Add created_at field
                    )
                    db.add(schedule_entry)
#                     print("complete recommendation_schedule")

            # Commit changes after processing all schedule entries for the day
            db.commit()

        
        print("finished_db")
        return recommendation_plan.recommendation_trip_id
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred while saving the itinerary: {str(e)}")




# Function to convert a list of items to a concatenated string based on the mapping
def convert_list_to_mapped_string(values: List[str], mapping: Dict[str, str]) -> str:
    mapped_values = [mapping.get(value) for value in values if mapping.get(value) is not None]
    if not mapped_values:
        raise ValueError(f"Invalid values provided: {values}")
    return ''.join(mapped_values)


# Function to convert RequestData to InputData
def convert_request_to_input(request_data: RequestData) -> InputData:
    # Convert the TRAVEL_PURPOSE list using the defined mapping
    if isinstance(request_data.TRAVEL_PURPOSE, list):
        travel_purpose_converted = convert_list_to_mapped_string(request_data.TRAVEL_PURPOSE, TRAVEL_PURPOSE_MAP)
    else:
        travel_purpose_converted = TRAVEL_PURPOSE_MAP.get(request_data.TRAVEL_PURPOSE)
        if travel_purpose_converted is None:
            raise ValueError(f"Invalid TRAVEL_PURPOSE value: {request_data.TRAVEL_PURPOSE}")
    
    # Convert the style using the defined mapping
    style_converted = TRAVEL_STYL_1_MAP.get(request_data.TRAVEL_STYL_1)
    if style_converted is None:
        raise ValueError(f"Invalid style value: {request_data.TRAVEL_STYL_1}")
    
    
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


def load_encoders():
    """Load encoder, scaler, and mlb files locally or fetch them from S3 if not available."""
    try:
        # Paths for the required files
        encoder_path_s3 = directory_name + "encoderAndScaler/encoder.pkl"
        scaler_path_s3 = directory_name + "encoderAndScaler/scaler.pkl"
        mlb_path_s3 = directory_name + "encoderAndScaler/mlb.pkl"

        # Local file paths
        local_directory = "./modeldata"
        os.makedirs(local_directory, exist_ok=True)  # Ensure the local directory exists
        encoder_path_local = os.path.join(local_directory, os.path.basename(encoder_path_s3))
        scaler_path_local = os.path.join(local_directory, os.path.basename(scaler_path_s3))
        mlb_path_local = os.path.join(local_directory, os.path.basename(mlb_path_s3))

        # Attempt to load locally if files exist
        encoders = {}
        for file_key, local_path, obj_name in [
            (encoder_path_s3, encoder_path_local, "encoder"),
            (scaler_path_s3, scaler_path_local, "scaler"),
            (mlb_path_s3, mlb_path_local, "mlb"),
        ]:
            if os.path.exists(local_path):
                logger.info(f"Local file {local_path} found. Loading...")
                with open(local_path, "rb") as f:
                    encoders[obj_name] = pickle.load(f)
            else:
                logger.info(f"Local file {local_path} not found. Downloading from S3...")
                s3.download_file(bucket_name, file_key, local_path)
                logger.info(f"{file_key} successfully downloaded and saved locally as {local_path}.")
                with open(local_path, "rb") as f:
                    encoders[obj_name] = pickle.load(f)

        return encoders

    except Exception as e:
        logger.error(f"Failed to load encoders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading encoders: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
