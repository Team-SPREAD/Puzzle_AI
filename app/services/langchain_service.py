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

    def generate_description(self, extracted_text: str) -> str:
        prompt = PromptTemplate(
            template=(
                "다음 텍스트가 이미지에서 추출되었습니다: {extracted_text}. "
                "이 텍스트를 기반으로 이미지를 설명하는 마크다운 문서를 작성하세요."
            ),
            input_variables=["extracted_text"],
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(extracted_text=extracted_text)
