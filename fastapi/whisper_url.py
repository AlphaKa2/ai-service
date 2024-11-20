from youtube_transcript_api import YouTubeTranscriptApi

def get_youtube_transcript(url):  # 매개변수 이름을 url로 변경
    # 비디오 ID 추출
    if "youtu.be" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0]
    else:
        video_id = url.split("v=")[-1].split("&")[0]
    try:
        # 자막 가져오기 (한국어로 설정)
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])

        # 자막 텍스트만 추출하여 새 리스트 생성
        script_only = [entry['text'] for entry in transcript]
        
        return script_only  # 스크립트만 포함된 리스트 반환
    except Exception as e:
        return str(e)  # 오류 메시지 반환
