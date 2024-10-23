from datetime import datetime
import json 
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import easyocr_url
import whisper_url
import youtube_info
import torch
import youtube_recommendation
from sqlalchemy.exc import IntegrityError
import database
from database import YoutubeVideo, RecommendationPlan, RecommendationPlace, RecommendationSchedule, RecommendedDay, RecommendationPlan
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import RedirectResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    ##allow_origins=["*"],  # 특정 출처를 지정할 수 있습니다. 모든 출처를 허용하려면 "*" 사용.
    allow_credentials=True,
    allow_methods=["*"],  # 모든 메소드를 허용하려면 "*" 사용.
    allow_headers=["*"],  # 모든 헤더를 허용하려면 "*" 사용.
)


# 데이터베이스 설정
DATABASE_URL = "mysql+mysqlconnector://user1:0000@14.35.173.14:40277/recommendation_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

@app.get("/recommendation-plan/{recommendation_trip_id}/")
async def get_recommendation_plan(recommendation_trip_id: int):
    db = SessionLocal()
    try:
        # 주어진 recommendation_trip_id로 추천 여행 계획 조회
        recommendation_plan = db.query(RecommendationPlan).filter_by(recommendation_trip_id=recommendation_trip_id).first()
        
        if recommendation_plan is None:
            raise HTTPException(status_code=404, detail="Recommendation plan not found.")
        
        return JSONResponse(content={"recommendation_trip_id": recommendation_plan.recommendation_trip_id, "title": recommendation_plan.title, "description": recommendation_plan.description})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        db.close()

@app.delete("/recommendation-plan/{recommendation_trip_id}/")
async def delete_recommendation_plan(recommendation_trip_id: int):
    db = SessionLocal()
    try:
        # 주어진 recommendation_trip_id로 추천 여행 계획 조회
        recommendation_plan = db.query(RecommendationPlan).filter_by(recommendation_trip_id=recommendation_trip_id).first()
        
        if recommendation_plan is None:
            raise HTTPException(status_code=404, detail="Recommendation plan not found.")
        
        # 추천 여행 계획 삭제
        db.delete(recommendation_plan)
        db.commit()
        
        return JSONResponse(content={"message": "Recommendation plan deleted successfully."})
    
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="Database integrity error: " + str(e.orig))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        db.close()

@app.post("/process-url/")
async def process_url(url: str, user_id: int):
    db = SessionLocal()
    try:
        # 기존 유튜브 영상 정보 조회
        existing_video = db.query(YoutubeVideo).filter_by(url=url).first()
        if existing_video:
            return JSONResponse(content={"travel_id": existing_video.recommendation_trip_id, "route": existing_video})

        # 유튜브 정보 저장
        youtube_info_data = youtube_info.extract_video_info(url)

        # EasyOCR 텍스트 추출
        easyocr_text = easyocr_url.easy_ocr_function(url)

        # Whisper 텍스트 추출
        whisper_text = whisper_url.get_youtube_transcript(url)
        if isinstance(whisper_text, list):
            whisper_text = "\n".join(whisper_text)

        # 여행 경로 생성
        travel_route_json = youtube_recommendation.generate_travel_route(
            easyocr_text,
            whisper_text,
            url,
            user_id,          
            db,          
            "api-key"  # API 키
        )

        return JSONResponse(content={"message": "Travel route and associated data saved successfully."})
        
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="Database integrity error: " + str(e.orig))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        db.close()


@app.get("/test-whisper/")
async def test_whisper(url: str):
    try:
        transcript = whisper_url.get_youtube_transcript(url)

        if isinstance(transcript, list):
            return JSONResponse(content={"message": "Transcription successful", "transcript": transcript})
        else:
            return JSONResponse(content={"message": "Transcription failed", "error": transcript})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-easyocr/")
async def test_easyocr(url: str):
    start_time = time.time()
    try:
        logging.info("Starting EasyOCR for URL: %s", url)
        
        easyocr_result_file = easyocr_url.easy_ocr_function(url)

        logging.info("EasyOCR function executed in %s seconds", time.time() - start_time)

        with open(easyocr_result_file, 'r', encoding='utf-8') as file:
            easyocr_text = file.read()

        logging.info("EasyOCR reading complete in %s seconds", time.time() - start_time)

        return JSONResponse(content={"message": "EasyOCR successful", "extracted_text": easyocr_text})
    
    except Exception as e:
        logging.error("Error during EasyOCR processing: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-youtube-info/")
async def test_youtube_info(url: str):
    
    try:
        youtube_info_data = youtube_info.extract_video_info(url)
        return JSONResponse(content={"message": "YouTube info retrieved successfully", "video_info": youtube_info_data})

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# JSON 파일에서 여행 경로 로드
def load_travel_route_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            travel_route_json = json.load(f)
        return travel_route_json
    except Exception as e:
        print(f"JSON 파일을 로드하는 중 오류 발생: {e}")
        return None

@app.post("/test-load-route/")
async def test_load_route(file_path: str):
    try:
        # JSON 파일에서 여행 경로를 로드
        travel_route_json = load_travel_route_from_file(file_path)
        if travel_route_json is None:
            raise HTTPException(status_code=400, detail="여행 경로를 로드하는 데 실패했습니다.")

        # 경로를 DB에 저장
        travel_id = youtube_recommendation.save_route_to_db(travel_route_json, db)

        return {"travel_id": travel_id, "route": travel_route_json}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

