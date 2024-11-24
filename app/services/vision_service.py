# app/services/vision_service.py
from google.cloud import vision

def extract_text_with_google_vision(image_bytes: bytes) -> str:
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(f"Google Vision API Error: {response.error.message}")
    return response.full_text_annotation.text

def refine_text_result(extracted_text: str) -> str:
    """
    Vision API로 추출된 텍스트를 정리하고 키워드를 강조합니다.
    """
    lines = extracted_text.split('\n')
    keywords = [line for line in lines if len(line) > 5]  # 간단한 필터링
    refined_text = "\n".join(keywords)
    return f"### 키워드 분석 결과\n{refined_text}"
