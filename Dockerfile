# 1. Python 베이스 이미지 설정 (Python 3.11 사용)
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 목록 복사
COPY requirements.txt .

# pip 및 Python 패키지 설치
RUN pip install --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. FastAPI 애플리케이션 실행 (uvicorn 사용)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
