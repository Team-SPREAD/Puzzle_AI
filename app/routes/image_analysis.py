from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel
from app.services.s3_service import fetch_image_from_s3, parse_s3_url
from app.services.vision_service import extract_text_with_google_vision, refine_text_result
from app.services.langchain_service import LangchainService

router = APIRouter()

# 요청 데이터 모델
class ImageAnalysisRequest(BaseModel):
    imageUrls: List[str]  # 이미지 URL 리스트

@router.post("/analyze-images")
async def analyze_images(request: ImageAnalysisRequest):
    valid_steps = [3, 4, 5, 6, 7, 8, 9]
    if len(request.imageUrls) != len(valid_steps):
        raise HTTPException(status_code=400, detail="3~9단계의 모든 이미지가 필요합니다.")

    step_descriptions = {
        3: "주제 선정 단계로, 사용자들이 다양한 의견을 제시하며 핵심 아이디어를 추출하는 초기 단계입니다.",
        4: "스프레드 단계로, 아이디어를 가지치기 방식으로 확장하며 창의성을 발휘하는 단계입니다.",
        5: "토론 단계로, 찬성, 반대, 중립 의견을 나누며 주요 쟁점을 정리합니다.",
        6: "페르소나 단계로, 다양한 사용자 관점을 분석하여 통찰을 제공합니다.",
        7: "문제 해결 단계로, 원인 분석과 해결 방안을 작성하며 실행 가능성을 평가합니다.",
        8: "사용자 스토리 단계로, 'Who', 'Goal', 'Action', 'Task'를 명확히 정의하여 사용자 관점을 명확히 합니다.",
        9: "메뉴 트리 단계로, 플로우차트를 분석하고 요구사항을 체계적으로 정리합니다.",
    }

    service = LangchainService()
    markdown_results = []

    # 1단계: 각 이미지에서 Markdown 생성
    for idx, image_url in enumerate(request.imageUrls):
        step_number = idx + 3
        try:
            bucket_name, image_key = parse_s3_url(image_url)
            image_bytes = fetch_image_from_s3(bucket_name, image_key)
            extracted_text = extract_text_with_google_vision(image_bytes)
            refined_text = refine_text_result(extracted_text)
            fixed_description = step_descriptions[step_number]
            first_result = service.generate_first_result(
                step_number=step_number,
                step_description=fixed_description,
                extracted_text=refined_text,
                image_url=image_url,
            )
            markdown_results.append(first_result)
        except Exception as e:
            markdown_results.append(
                f"### {step_number}단계\nError processing image at {image_url}: {str(e)}"
            )

    # Markdown 결과를 결합
    combined_markdown = "\n\n".join(markdown_results)

    # 2단계: 프로젝트 기획서 생성
    try:
        project_plan = service.generate_second_result(markdown_content=combined_markdown)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating project plan: {str(e)}")

    # 3단계: 요구사항 명세서 추가
    try:
        requirements_spec = service.generate_third_result(markdown_content=combined_markdown)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating requirements spec: {str(e)}")

    # 최종 결과 반환 (Markdown + 프로젝트 기획서 + 요구사항 명세서)
    final_result = f"{combined_markdown}\n\n# 프로젝트 기획서\n\n{project_plan}\n\n# 요구사항 명세서\n\n{requirements_spec}"

    return {"result": final_result}
