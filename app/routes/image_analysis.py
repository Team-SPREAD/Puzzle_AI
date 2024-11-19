from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel
from app.services.s3_service import fetch_image_from_s3, parse_s3_url
from app.services.vision_service import extract_text_with_google_vision
from app.services.langchain_service import LangchainService

router = APIRouter()

# 요청 데이터 모델
class ImageAnalysisRequest(BaseModel):
    imageUrls: List[str]  # 이미지 URL 리스트

@router.post("/analyze-images")
async def analyze_images(request: ImageAnalysisRequest):

    # 유효한 단계 정의 (3~9)
    valid_steps = [3, 4, 5, 6, 7, 8, 9]
    if len(request.imageUrls) != len(valid_steps):
        raise HTTPException(status_code=400, detail="3~9단계의 모든 이미지가 필요합니다.")

    # 단계별 고정된 설명 정의
    step_descriptions = {
        3: "이 단계는 주제선정 단계입니다.",
        4: "이 단계는 스프레드 단계 3단계에서 정해진 주제에서 아이디어를 확장하는 단계입니다.",
        5: "이 단계는 토론하는 단계입니다.",
        6: "이 단계는 페르소나 단계입니다.",
        7: "이 단계는 문제해결 단계입니다.",
        8: "이 단계는 사용자 스토리 단계입니다.",
        9: "이 단계는 메뉴트리 단계입니다. 메뉴트리는 플로우차트를 이용합니다.",
    }

    service = LangchainService()  # Langchain 서비스 초기화
    markdown_results = []

    # 이미지와 설명을 기반으로 각 단계 처리
    for idx, image_url in enumerate(request.imageUrls):
        step_number = idx + 3  # 3단계부터 시작
        try:
            # S3에서 이미지 가져오기
            bucket_name, image_key = parse_s3_url(image_url)
            image_bytes = fetch_image_from_s3(bucket_name, image_key)

            # Vision API로 텍스트 추출
            extracted_text = extract_text_with_google_vision(image_bytes)

            # 단계별 설명과 결과를 조합
            fixed_description = step_descriptions.get(step_number, "단계 설명이 없습니다.")
            markdown_result = service.generate_description(
                step_number=step_number,
                step_description=fixed_description,
                extracted_text=extracted_text,
                image_url=image_url,
            )
            markdown_results.append(f"### {step_number}단계\n{markdown_result}")

        except Exception as e:
            markdown_results.append(f"### {step_number}단계\nError processing image at {image_url}: {str(e)}")

    # 최종 Markdown 결과 반환
    final_markdown = "\n\n".join(markdown_results)

    if not markdown_results:
        raise HTTPException(status_code=400, detail="No images could be processed.")

    return {"result": final_markdown}
