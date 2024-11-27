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

        # 3단계: 요구사항 명세서 추가
        self.third_prompt_template = PromptTemplate(
    template=(
        "{markdown_content}\n\n"
        "위 내용을 그대로 복사하고 여기에 요구사항 명세서도 추가해 주세요.\n\n"
        "**요구사항 명세서**:\n"
        "1. 프로젝트를 구현하기 위한 주요 요구사항을 정의하세요.\n"
        "2. 사용자 기능, 시스템 기능, 기술 스택 등을 포함하세요.\n"
        "3. 최종 결과는 Markdown 형식으로 작성해주세요.\n\n"
        "예시:\n"
        "# 요구사항 명세서\n\n"
        "## 1. 사용자 계정\n"
        "- 회원가입: 이메일, 비밀번호, 닉네임 입력\n"
        "- 로그인: 이메일/비밀번호로 로그인, 소셜 로그인 지원\n"
        "- 비밀번호 찾기: 비밀번호 재설정 메일 발송\n\n"
        "## 2. 물품 등록 및 관리\n"
        "- 물품 등록: 제목, 설명, 가격, 대여 기간 입력\n"
        "- 물품 수정: 기존 등록 내용 수정\n"
        "- 물품 삭제: 더 이상 대여하지 않을 경우 삭제\n\n"
        "## 기술 스택\n"
        "- 프론트엔드: React\n"
        "- 백엔드: NestJS\n"
        "- 데이터베이스: MongoDB\n"
        "- 배포 환경: AWS EC2, Docker"
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

    def generate_third_result(self, markdown_content):
        """3단계: 요구사항 명세서 생성"""
        chain = LLMChain(llm=self.llm, prompt=self.third_prompt_template)
        return chain.run(markdown_content=markdown_content)