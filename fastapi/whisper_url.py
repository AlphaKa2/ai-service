from youtube_transcript_api import YouTubeTranscriptApi

def get_youtube_transcript(url):  # 매개변수 이름을 url로 유지
    # 쿠키 파일 경로 (현재 디렉토리의 cookies.txt로 설정)
    cookie_file = "cookies.txt"

    # 비디오 ID 추출
    if "youtu.be" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0]
    else:
        video_id = url.split("v=")[-1].split("&")[0]

    try:
        # 자막 가져오기 (한국어 및 영어 우선)
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'], cookies=cookie_file)

        # 자막 텍스트만 추출하여 새 리스트 생성
        script_only = [entry['text'] for entry in transcript]
        
        return script_only  # 스크립트만 포함된 리스트 반환
    except Exception as e:
        return str(e)  # 오류 메시지 반환

# 테스트 코드
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=eMpsJz5n-Mg"  # 테스트할 YouTube URL

    # get_youtube_transcript 함수 호출
    transcript = get_youtube_transcript(youtube_url)

    # 자막 텍스트 출력
    if isinstance(transcript, list):
        print("자막 텍스트:")
        for line in transcript:
            print(line)
    else:
        print("오류 발생:", transcript)  # 오류 메시지 출력
