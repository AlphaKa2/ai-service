import json
import logging
from datetime import datetime
from yt_dlp import YoutubeDL
import requests

logging.basicConfig(level=logging.INFO)

OUTPUT_JSON = 'youtube_data.json'

def extract_video_info(youtube_url):
    ydl_opts = {
        'format': 'best',
        "cookies": "./cookies.txt",  # Use cookies for authentication
    }

    current_time = datetime.now().isoformat()  # datetime을 ISO 형식의 문자열로 변환

    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        
        video_info = {
            "video_id": info_dict.get("id"),
            "youtuber_id": info_dict.get("channel_id"),
            "title": info_dict.get("title"),
            "url": info_dict.get("webpage_url"),
            "series_id": None,
            "series_order": None,
            "thumbnail_url": info_dict.get("thumbnail"),
            "created_at": current_time,
            "updated_at": current_time             
        }
        
        # 유튜버 정보 추가
        youtuber_info = {
            "youtuber_id": info_dict.get("channel_id"),
            "name": info_dict.get("channel_name"),
            "url": info_dict.get("channel_url"),
            "created_at": current_time,
            "updated_at": current_time         
        }
        
        return {"video_info": video_info, "youtuber_info": youtuber_info}

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
        # 유튜브 영상 정보를 추출합니다.
        video_data = extract_video_info(youtube_url)
        
        # JSON 파일로 저장합니다.
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(video_data, f, ensure_ascii=False, indent=4)

        logging.info(f"데이터가 {output_file}에 성공적으로 저장되었습니다.")

    except Exception as e:
        logging.error(f"에러 발생: {e}")

# 아래의 코드는 이 모듈이 직접 실행될 때만 동작합니다.
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=eMpsJz5n-Mg"  # 테스트할 유튜브 URL
    travel_id = 1  # 테스트용 travel_id
    save_to_json(youtube_url, OUTPUT_JSON, travel_id)