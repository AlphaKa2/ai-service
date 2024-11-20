from youtube_transcript_api import YouTubeTranscriptApi
import easyocr
import os
import cv2
import shutil
import numpy as np
from PIL import Image
import yt_dlp
import re
import logging
from Levenshtein import distance as levenshtein_distance
import uuid  # 고유한 ID 생성을 위한 모듈

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 이미지 폴더 설정하기
def set_image_folder():
    if os.path.exists('image_frames'):
        shutil.rmtree('image_frames')
    os.mkdir('image_frames')

# 유튜브 비디오 다운로드 함수
def download_youtube_video(youtube_url):
    video_file_name = f'downloaded_video_{uuid.uuid4()}.mp4'  # 고유한 비디오 파일명 생성
    ydl_opts = {
        'format': 'best',
        'outtmpl': video_file_name,
        'quiet': False,
        'subtitleslangs': ['a.en', 'a.ko', 'a.fr', 'a.de'],  # 자동 생성 자막 언어 추가
        'writesubtitles': True,  # 자막 파일 다운로드
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            logging.info("Downloading video from URL: %s", youtube_url)
            ydl.download([youtube_url])
            subtitle_file_name = video_file_name.rsplit('.', 1)[0] + '.vtt'  # 자막 파일 이름 생성
            return video_file_name, subtitle_file_name
        except Exception as e:
            logging.error("Error downloading video: %s", e)
            return None, None

# 텍스트 정제 함수 (공백 제거)
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9가-힣\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else None  # 빈 텍스트는 None으로 반환

# 편집 거리 기반 유사도 비교 함수
def is_similar_with_levenshtein(text1, text2, threshold=0.7):
    if not text1 or not text2:
        return False
    edit_distance = levenshtein_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    similarity = 1 - (edit_distance / max_len)
    return similarity > threshold

# 프레임 저장과 텍스트 추출을 동시에 진행하는 함수
def extract_images_and_text(video, fps, language):
    reader = easyocr.Reader([language], gpu=True)
    frame_count = 0
    extracted_texts = []
    success = True

    while success:
        success, image = video.read()
        if success and frame_count % int(fps) == 0:
            frame_file = f'image_frames/frame_{frame_count}.png'
            cv2.imwrite(frame_file, image)
            logging.info("Extracted frame: %s", frame_count)

            # 이미지에서 텍스트 추출
            image_pil = Image.open(frame_file)
            try:
                text = reader.readtext(np.array(image_pil), detail=0, paragraph=True)
                joined_text = clean_text(' '.join(text))

                # 중복 및 공백 필터링
                if joined_text:
                    if extracted_texts and is_similar_with_levenshtein(joined_text, extracted_texts[-1][0]):
                        logging.info("Similar text detected for frame %s, skipping.", frame_count)
                    else:
                        extracted_texts.append((joined_text, frame_count))
                        logging.info("Extracted text from frame %s: %s", frame_count, joined_text)
            except Exception as e:
                logging.error("Error processing frame %s: %s", frame_count, e)
        
        frame_count += 1

    return '\n'.join(text for text, _ in extracted_texts)

def easy_ocr_function(youtube_url):
    set_image_folder()
    video_path, subtitle_path = download_youtube_video(youtube_url)  # URL로 비디오 다운로드
    if not video_path:
        logging.error("Video download failed")
        return None

    detected_language = 'ko'  # 테스트 목적으로 기본값으로 한국어 설정
    video, fps = cv2.VideoCapture(video_path), 30  # 비디오와 FPS 설정
    extracted_text = extract_images_and_text(video, fps, detected_language)

    with open('extracted_text_easyocr_url.txt', 'w', encoding='utf-8') as f:
        f.write(extracted_text)
        logging.info("Extracted text saved to 'extracted_text_easyocr_url.txt'")

    return 'extracted_text_easyocr_url.txt'

if __name__ == "__main__":
    youtube_url = "https://youtu.be/LOuxNoCkh8A"  # 미리 정의된 URL
    output_file = easy_ocr_function(youtube_url)
    if output_file:
        print(f"추출된 텍스트가 {output_file}에 저장되었습니다.")