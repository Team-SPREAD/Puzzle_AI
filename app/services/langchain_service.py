import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel
from app.services.s3_service import fetch_image_from_s3, parse_s3_url
from app.services.vision_service import extract_text_with_google_vision

# FastAPI 라우터
router = APIRouter()

# 요청 데이터 모델
class ImageAnalysisRequest(BaseModel):
    imageUrls: List[str]  # 이미지 URL 리스트


# LangChain 서비스 정의
class LangchainService:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.5,
        )

        # 1단계: Markdown 생성 프롬프트
        self.first_prompt_template = PromptTemplate(
            template=(
                "아래 정보를 활용하여 Markdown 문서를 작성하세요:\n\n"
                "**설명**:\n{step_description}\n\n"
                "**이미지 분석 결과**:\n{extracted_text}\n\n"
                "**이미지 URL**: {image_url}\n\n"
                "요구사항:\n"
                "1. 내용을 요약하고 Markdown의 헤더, 리스트, 표 등을 활용하여 보기 좋게 구성하세요.\n"
                "2. 이미지 분석 결과에서 중요한 정보를 강조하세요.\n"
                "3. 간결하면서도 전문적으로 작성하세요.\n\n"
                "예시:\n"
                "# 3단계: 주제 선정\n\n"
                "### 분석 결과\n"
                "- **의견 1**: xxx\n"
                "- **의견 2**: xxx\n\n"
                "![이미지](image_url)\n"
            ),
            input_variables=["step_number", "step_description", "extracted_text", "image_url"],
        )

        # 2단계: 프로젝트 기획서 생성 프롬프트
        self.second_prompt_template = PromptTemplate(
            template=(
                "{markdown_content}\n\n"
                "위 내용을 기반으로 프로젝트 기획서를 작성하세요.\n\n"
                "**요구사항**:\n"
                "1. 각 단계별로 내용을 정리하고 이미지와 연결되는 정보를 잘 보여주세요.\n"
                "2. 각 단계의 주요 아이디어를 요약하며 시각적 요소(이미지 설명)를 추가하세요.\n"
                "3. 최종 결과는 Markdown 형식으로 이쁘게 꾸며주세요.\n"
                "4. 단계별 주요 정보를 분명히 드러내도록 작성하세요.\n\n"
                "예시:\n"
                "# 프로젝트 기획서\n\n"
                "## 3단계: 주제 선정\n\n"
                "### 주요 내용\n"
                "- 의견: xxx\n"
                "- 분석 결과: xxx\n\n"
                "![이미지](image_url)\n\n"
                "각 단계가 명확히 구분되도록 작성해주세요."
            ),
            input_variables=["markdown_content"],
        )

    def generate_first_result(self, step_number, step_description, extracted_text, image_url):
        """1단계: Markdown 생성"""
        chain = LLMChain(llm=self.llm, prompt=self.first_prompt_template)
        return chain.run(
            step_number=step_number,
            step_description=step_description,
            extracted_text=extracted_text,
            image_url=image_url,
        )

    def generate_second_result(self, markdown_content):
        """2단계: 최종 기획서 생성"""
        chain = LLMChain(llm=self.llm, prompt=self.second_prompt_template)
        return chain.run(markdown_content=markdown_content)


@router.post("/analyze-images")
async def analyze_images(request: ImageAnalysisRequest):
    # 유효한 단계 정의 (3~9)
    valid_steps = [3, 4, 5, 6, 7, 8, 9]
    if len(request.imageUrls) != len(valid_steps):
        raise HTTPException(status_code=400, detail="3~9단계의 모든 이미지가 필요합니다.")

    # 단계별 고정된 설명 정의
    step_descriptions = {
        3: "주제선정 단계로, 사용자들이 다양한 의견을 제시하는 초기 단계입니다.",
        4: "스프레드 단계로, 주제에서 다양한 아이디어가 가지치기 방식으로 확장되는 단계입니다.",
        5: "토론 단계로, 찬성, 반대, 중립 등 다양한 의견이 교환되는 단계입니다.",
        6: "페르소나 단계로, 다양한 시각에서 의견을 정리합니다.",
        7: "문제해결 단계로, 상황 정의, 원인 분석, 해결 방안을 작성합니다.",
        8: "사용자 스토리 단계로, 'Who', 'Goal', 'Action', 'Task'를 분석합니다.",
        9: "메뉴트리 단계로, 플로우차트를 분석하고 요구사항을 명세합니다.",
    }

    # LangChain 서비스 초기화
    service = LangchainService()
    markdown_results = []

    # 각 단계에 대해 Markdown 생성
    for idx, image_url in enumerate(request.imageUrls):
        step_number = idx + 3
        try:
            # S3에서 이미지 가져오기
            bucket_name, image_key = parse_s3_url(image_url)
            image_bytes = fetch_image_from_s3(bucket_name, image_key)

            # Vision API로 텍스트 추출
            extracted_text = extract_text_with_google_vision(image_bytes)

            # 1단계: Markdown 생성
            step_description = step_descriptions.get(step_number, "단계 설명이 없습니다.")
            markdown = service.generate_first_result(
                step_number=step_number,
                step_description=step_description,
                extracted_text=extracted_text,
                image_url=image_url,
            )
            markdown_results.append(markdown)

        except Exception as e:
            markdown_results.append(
                f"### {step_number}단계\nError processing image at {image_url}: {str(e)}"
            )

    # Markdown 결과 조합
    combined_markdown = "\n\n".join(markdown_results)

    # 2단계: 최종 기획서 생성
    try:
        final_result = service.generate_second_result(markdown_content=combined_markdown)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating final result: {str(e)}")

    # 최종 결과 반환
    return {"result": final_result}
