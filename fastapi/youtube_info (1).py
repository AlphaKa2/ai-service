import json  
from datetime import datetime
import logging
import requests
from sqlalchemy.orm import Session
from yt_dlp import YoutubeDL
from database import Youtuber, YoutubeVideo, SessionLocal

logging.basicConfig(level=logging.INFO)

OUTPUT_JSON = 'youtube_data.json'

def extract_video_info(youtube_url):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
    }

    current_time = datetime.now()  # current_time을 여기서 정의합니다.

    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        current_time = datetime.now().isoformat()  # datetime을 ISO 형식의 문자열로 변환
        video_info = {
            "video_id": info_dict.get("id"),
            "youtuber_id": info_dict.get("channel_id"),  # video_info["channel_id"] -> info_dict.get("channel_id")로 수정
            "title": info_dict.get("title"),
            "url": info_dict.get("webpage_url"),
            "series_id": None,
            "series_order": None,
            "thumbnail_url": info_dict.get("thumbnail"),
            "created_at": current_time,
            "updated_at": current_time             
        }
        
        # 데이터베이스에 저장
        structure_and_save_data(video_info)
        
        # 유튜버 정보 추가
        youtuber_info = {
            "youtuber_id": info_dict.get("channel_id"),  # video_info["channel_id"] -> info_dict.get("channel_id")로 수정
            "name": info_dict.get("channel_name"),  # video_info["channel_name"] -> info_dict.get("channel_name")로 수정
            "url": info_dict.get("channel_url"),  # video_info["channel_url"] -> info_dict.get("channel_url")로 수정
            "created_at": current_time,
            "updated_at": current_time         
        }
        
        return {"video_info": video_info, "youtuber_info": youtuber_info}

def structure_and_save_data(video_info):
    current_time = datetime.now()  # current_time을 여기서 다시 정의합니다.
    db: Session = SessionLocal()

    try:
        youtuber = db.query(Youtuber).filter_by(youtuber_id=video_info["youtuber_id"]).first()  # channel_id -> youtuber_id로 수정
        if not youtuber:
            youtuber = Youtuber(
                youtuber_id=video_info["youtuber_id"],  # video_info["channel_id"] -> video_info["youtuber_id"]로 수정
                name=video_info.get("title"),  # video_info["channel_name"] -> video_info.get("title")로 수정
                url=video_info.get("url"),  # video_info["channel_url"] -> video_info.get("url")로 수정
                created_at=current_time,
                updated_at=current_time
            )
            db.add(youtuber)
        else:
            youtuber.name = video_info.get("title")  # video_info["channel_name"] -> video_info.get("title")로 수정
            youtuber.url = video_info.get("url")  # video_info["channel_url"] -> video_info.get("url")로 수정
            youtuber.updated_at = current_time

        video = db.query(YoutubeVideo).filter_by(video_id=video_info["video_id"]).first()
        if not video:
            video = YoutubeVideo(
                video_id=video_info["video_id"],
                youtuber_id=youtuber.youtuber_id,
                title=video_info["title"],
                url=video_info["url"],  # video_info["video_url"] -> video_info["url"]로 수정
                series_id=video_info.get("series_id"),
                series_order=video_info.get("series_order"),
                created_at=current_time,
                updated_at=current_time,
                thumbnail_url=video_info["thumbnail_url"]
            )
            db.add(video)
        else:
            video.title = video_info["title"]
            video.url = video_info["url"]  # video_info["video_url"] -> video_info["url"]로 수정
            video.series_id = video_info.get("series_id")
            video.series_order = video_info.get("series_order")
            video.updated_at = current_time
            video.thumbnail_url = video_info["thumbnail_url"]

        db.commit()
        db.refresh(youtuber)
        db.refresh(video)

    except Exception as e:
        db.rollback()
        raise ValueError(f"데이터베이스 저장 오류: {e}")

def get_high_resolution_thumbnail(video_id):
    thumbnail_urls = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/sddefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/default.jpg"
    ]

    for url in thumbnail_urls:
        try:
            response = requests.head(url)
            if response.status_code == 200:
                return url
        except requests.RequestException:
            continue

    return ""

def save_to_json(youtube_url, output_file, travel_id):
    try:
        video_data = extract_video_info(youtube_url)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(video_data, f, ensure_ascii=False, indent=4)

    except ValueError as e:
        logging.error(e)

# 아래의 코드는 이 모듈이 직접 실행될 때만 동작합니다.
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=example_video_id"
    travel_id = 1
    save_to_json(youtube_url, OUTPUT_JSON, travel_id)
