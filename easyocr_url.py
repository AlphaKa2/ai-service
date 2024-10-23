import easyocr
import os
import cv2
import shutil
import numpy as np
from PIL import Image
import yt_dlp
import re
import difflib
import logging  # 로깅을 위한 모듈 추가

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # 포맷 추가

# 이미지 폴더 설정하기
def set_image_folder():
    if os.path.exists('image_frames'):
        shutil.rmtree('image_frames')
    os.mkdir('image_frames')

# 비디오 읽기
def read_video(video_path):
    logging.info("Reading video: %s", video_path)  # 비디오 읽기 로그
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    logging.info("Video FPS: %s, Total Frames: %s", fps, total_frames)  # FPS 및 총 프레임 로그
    return video, fps

# 이미지 추출하기 (프레임별 시간 계산)
def extract_images(video, fps):
    frame_count = 0
    success = True
    while success:
        success, image = video.read()
        if success and frame_count % (1 * int(fps)) == 0:  # 1초마다 한 번씩 프레임 저장
            cv2.imwrite(f'image_frames/frame_{frame_count}.png', image)
            logging.info("Extracted frame: %s", frame_count)  # 프레임 추출 로그
        frame_count += 1

# 텍스트 정제 함수: 특수 문자 제거 및 공백 정리
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9가-힣\s]', '', text)  # 특수 문자 제거
    return re.sub(r'\s+', ' ', text).strip()  # 다중 공백 제거

# 텍스트 유사도 비교 함수
def is_similar(text1, text2, threshold=0.80):
    return difflib.SequenceMatcher(None, text1, text2).ratio() > threshold

# 이미지 텍스트 추출하기
def extract_text():
    logging.info("Starting text extraction from images")  # 텍스트 추출 시작 로그
    reader = easyocr.Reader((['ko']), gpu=True)  # EasyOCR 모델 로드
    image_files = sorted(os.listdir('image_frames'))  # 이미지 파일 정렬
    extracted_texts = []
    
    for image_file in image_files:
        logging.info("Processing image: %s", image_file)  # 현재 처리 중인 이미지 로그
        image = Image.open(f'image_frames/{image_file}')
        try:
            text = reader.readtext(np.array(image), detail=0, paragraph=True)  # EasyOCR을 사용하여 텍스트 추출
            if not text:  # 텍스트가 비어있을 경우
                logging.warning("No text extracted from %s", image_file)  # 경고 로그
                continue
            joined_text = clean_text(' '.join(text))  # 추출된 텍스트 정제 및 하나의 문자열로 결합

            # 빈 텍스트가 아니고, 유사한 텍스트가 이미 리스트에 없을 경우 저장
            if joined_text and not any(is_similar(joined_text, existing_text) for existing_text in extracted_texts):
                extracted_texts.append(joined_text)
                logging.info("Extracted text from %s: %s", image_file, joined_text)  # 추출된 텍스트 로그
            else:
                logging.info("Similar text already exists for %s, skipping.", image_file)  # 유사한 텍스트 경고 로그
        except Exception as e:
            logging.error("Error processing image %s: %s", image_file, e)  # 이미지 처리 중 오류 로그

    return ' '.join(extracted_texts)

# 비디오 다운로드
def download_youtube_video(youtube_url):
    logging.info("Downloading video from URL: %s", youtube_url)  # 비디오 다운로드 로그
    try:
        ydl_opts = {
            'format': 'best',  # 최상의 비디오 형식 다운로드
            'outtmpl': 'downloads/%(title)s.%(ext)s',  # 저장할 파일 경로
            'merge_output_format': 'mp4'  # 병합할 형식 지정
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            logging.info("Downloaded video: %s", ydl.prepare_filename(info_dict))  # 다운로드 완료 로그
            return ydl.prepare_filename(info_dict)
    except Exception as e:
        logging.error("An error occurred during video download: %s", e)  # 오류 로그
        return None

# 메인 함수
def easy_ocr_function(youtube_url):
    set_image_folder()
    video_path = download_youtube_video(youtube_url)
    if not video_path:
        logging.error("Video download failed")  # 다운로드 실패 로그
        return None

    video, fps = read_video(video_path)
    extract_images(video, fps)
    extracted_text = extract_text()

    # 텍스트 저장하기
    with open('extracted_text_easyocr_url.txt', 'w', encoding='utf-8') as f:
        f.write(extracted_text)
        logging.info("Extracted text saved to 'extracted_text_easyocr_url.txt'")  # 텍스트 저장 로그
    
    return 'extracted_text_easyocr_url.txt'
