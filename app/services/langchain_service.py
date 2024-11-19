import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

class LangchainService:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
        )
        self.prompt_template = PromptTemplate(
            template=(
                "다음 텍스트를 기반으로 Markdown 문서를 작성하세요:\n"
                "- 단계: {step_number}\n"
                "- 설명: {step_description}\n"
                "- 이미지 분석 결과: {extracted_text}\n"
                "- 이미지 URL: {image_url}\n"
                "\nMarkdown 결과를 반환하세요."
            ),
            input_variables=["step_number", "step_description", "extracted_text", "image_url"],
        )

    def generate_description(self, step_number, step_description, extracted_text, image_url):
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        return chain.run(
            step_number=step_number,
            step_description=step_description,
            extracted_text=extracted_text,
            image_url=image_url,
        )

