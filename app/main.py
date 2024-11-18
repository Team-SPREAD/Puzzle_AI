import os
from fastapi import FastAPI
from dotenv import load_dotenv
from app.routes.image_analysis import router as image_router

# .env 파일 로드
load_dotenv()

app = FastAPI()

# 라우터 등록
app.include_router(image_router)

@app.get("/")
def read_root():
    return {"message": "S3 이미지 분석 API 작동 중"}
