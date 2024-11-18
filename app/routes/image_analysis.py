from fastapi import APIRouter, HTTPException
from app.services.s3_service import fetch_image_from_s3, parse_s3_url
from app.services.vision_service import extract_text_with_google_vision
from app.services.langchain_service import LangchainService
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# 요청 데이터 모델
class ImageRequest(BaseModel):
    s3_url: Optional[str] = None
    bucket_name: Optional[str] = None
    image_key: Optional[str] = None

@router.post("/analyze-image")
def analyze_image(request: ImageRequest):
    try:
        # S3 URL 파싱
        if request.s3_url:
            bucket_name, image_key = parse_s3_url(request.s3_url)  # 독립 함수로 직접 호출
        else:
            bucket_name, image_key = request.bucket_name, request.image_key

        # S3에서 이미지 가져오기
        image_bytes = fetch_image_from_s3(bucket_name, image_key)

        # Google Vision API로 텍스트 추출
        extracted_text = extract_text_with_google_vision(image_bytes)

        # LangChain을 사용하여 마크다운 설명 생성
        service = LangchainService()
        description = service.generate_description(extracted_text)

        return {"extracted_text": extracted_text, "description": description}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
