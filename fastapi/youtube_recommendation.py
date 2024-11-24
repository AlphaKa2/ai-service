from datetime import datetime
import logging
import os
import json
from fastapi import HTTPException
import requests
from bs4 import BeautifulSoup
from database import RecommendationPlan, RecommendationDay, RecommendationPlace, RecommendationSchedule, TravelPlanYoutubeVideo, YoutubeVideo
from sqlalchemy.orm import Session

# Whisper 텍스트 읽기 함수
def load_whisper_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            whisper_text = f.readlines()
        return [line.strip() for line in whisper_text if line.strip()]  # 빈 줄 제거
    except Exception as e:
        print(f"Whisper 텍스트 파일을 로드하는 중 오류 발생: {e}")
        return []

# OCR 텍스트 읽기 함수
def load_ocr_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ocr_text = f.readlines()
        return [line.strip() for line in ocr_text if line.strip()]  # 빈 줄 제거
    except Exception as e:
        print(f"OCR 텍스트 파일을 로드하는 중 오류 발생: {e}")
        return []

def get_youtube_title(video_url):
    try:
        response = requests.get(video_url)
        response.raise_for_status()  # 응답 코드가 200이 아닐 경우 예외 발생
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('title')
        
        if title_tag:
            full_title = title_tag.text.strip()
            title = full_title.replace(" - YouTube", "") if " - YouTube" in full_title else full_title
            return title
        else:
            raise ValueError("제목을 찾을 수 없습니다.")
            
    except requests.RequestException as req_err:
        print(f"HTTP 요청 중 오류 발생: {req_err}")
        return None
    except Exception as e:
        print(f"유튜브 제목을 가져오는 중 오류 발생: {e}")
        return None


# ChatGPT API 호출 함수
def call_chatgpt_api(prompt, api_key):
    try:
        headers = {
            'Authorization': f"key",
            'Content-Type': 'application/json',
        }
        data = {
            'model': 'gpt-4-turbo',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 4096,
            'temperature': 0.7,
        }
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        print("API 응답:", response_data)


        if 'error' in response_data:
            print("Error:", response_data['error']['message'])
            return None

        return response_data['choices'][0]['message']['content'] if 'choices' in response_data else None
    except Exception as e:
        print(f"ChatGPT API 호출 중 오류 발생: {e}")
        return None

# 여행 경로 생성 프롬프트 함수
def generate_travel_route_prompt(texts, title):
    texts_str = "\n".join(texts)
    prompt = (
        f"비디오 제목 '{title}'의 노트는 easyocr로 추출한 텍스트와 유튜브 스크립트로 추출한 텍스트를 합친 텍스트이다. 이를 기반으로 포괄적인 여행 경로를 순서대로 생성해 주세요:\n"
        f"{texts_str}\n\n"
        "여행 경로는 하루 단위로 구분하고, 각 장소에 대한 간략한 설명을 포함해 주세요."
    )
    return prompt

# JSON 변환 프롬프트 함수
def generate_json_prompt(route_text, title, description):
    prompt = f"""
아래의 여행 경로를 바탕으로, 지정된 형식에 맞는 JSON 객체를 생성해 주세요. JSON은 정확히 아래 형식을 따라야 합니다:

{{
    "title": "{title}",
    "description": "{description}",
    "days": [
        {{
            "day": 1,
            "schedule": [
                {{
                    "place": "장소 이름",
                    "content": "장소에 대한 설명."
                }}
            ]
        }}
    ]
}}

여행 경로는 다음과 같습니다:

{route_text}

각 일자별로 "day"와 "schedule"을 포함하고, "schedule"에는 "place"와 "content" 필드를 적절히 채워 넣어 주세요. 최대한 스크립트의 장소를 모두 넣어주고 장소 이름에 오타가 있으면 수정해줘 "어느집","바닷가"이런 장소는 웬만하면 쓰지말고 최대한 title과 영상 내용으로 장소를 추론해줘.. 응답은 순수한 JSON만 반환해 주세요. 코드 블록(```json ... ```)을 포함하지 마세요.
"""
    return prompt

def generate_json(route_text, title, description, api_key):
    prompt = generate_json_prompt(route_text, title, description)
    json_response = call_chatgpt_api(prompt, api_key)

    # JSON 응답이 None인 경우 처리
    if json_response is None:
        print("ChatGPT API에서 유효한 응답을 받지 못했습니다.")
        return None

    if isinstance(json_response, str):
        try:
            # 응답이 문자열인 경우 JSON 객체로 변환
            travel_route_json = json.loads(json_response)
            print("응답이 문자열에서 JSON 객체로 변환되었습니다.")
            return travel_route_json
        except json.JSONDecodeError:
            print("응답 문자열을 JSON으로 변환하는 데 실패했습니다.")
            return None

# 여행 경로에서 데이터 저장 함수
def save_route_to_db(travel_route_json, db, user_id):
    # 1. RecommendationPlan 저장
    recommendation_plan = RecommendationPlan(
        user_id=user_id,
        name=travel_route_json["title"],
        description=travel_route_json["description"],
        recommendation_type='YOUTUBER_FOLLOW'  # 추천 타입은 적절히 설정
    )
    db.add(recommendation_plan)
    db.flush()  # ID를 가져오기 위해 flush()

    # 2. RecommendedDay 및 RecommendationSchedule 저장
    for day in travel_route_json["days"]:
        recommended_day = RecommendationDay(
            recommended_trip_id=recommendation_plan.recommendation_trip_id,
            day_number=day["day"],
            date=datetime.now()  # 적절한 날짜를 설정
        )
        db.add(recommended_day)
        db.flush()  # ID를 가져오기 위해 flush()

        # 3. 각 day's schedule 저장
        for idx, schedule in enumerate(day["schedule"]):
            # RecommendationPlace 저장
            recommendation_place = RecommendationPlace(
                place_name=schedule["place"],
                content=schedule["content"]
            )
            db.add(recommendation_place)
            db.flush()  # ID를 가져오기 위해 flush()

            # RecommendationSchedule 저장
            recommendation_schedule = RecommendationSchedule(
                day_id=recommended_day.day_id,
                place_id=recommendation_place.place_id,
                recommendation_trip_id=recommendation_plan.recommendation_trip_id,
                schedule_order=idx + 1  # 1부터 시작
            )
            db.add(recommendation_schedule)
            # TravelPlanYoutubeVideo 저장
        for video in recommendation_plan.youtube_videos:
            travel_plan_youtube_video = TravelPlanYoutubeVideo(
                travel_id=recommendation_plan.recommendation_trip_id,
                video_id=video.video_id,  # 여기서 video.video_id를 사용
                place_id=recommendation_place.place_id  # 외래 키
            )
            db.add(travel_plan_youtube_video)

    # 최종 커밋
    db.commit()

    return recommendation_plan.recommendation_trip_id



# 메인 경로 생성 함수
def generate_travel_route(easyocr_text, whisper_text, youtube_url,user_id, db: Session, api_key: str):
    title = get_youtube_title(youtube_url)

    if title is None:
        raise ValueError("유튜브 제목을 가져오는 데 실패했습니다.")
    
    # 프롬프트 생성 및 ChatGPT API 호출
    combined_texts = easyocr_text + whisper_text
    travel_route_text = generate_travel_route_prompt(combined_texts, title)
    travel_route_json = generate_json(travel_route_text, title, f"{title} 기반 여행 경로", api_key)
    
    # JSON 응답이 None인지 확인
    if travel_route_json is None:
        raise ValueError("여행 경로 JSON 생성에 실패했습니다.")

    # 경로를 DB에 저장
    travel_id = save_route_to_db(travel_route_json, db, user_id)
    print(travel_route_json)
    return travel_route_json

