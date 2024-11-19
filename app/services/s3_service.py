import os
import boto3
from urllib.parse import urlparse

def parse_s3_url(s3_url: str):
    """
    S3 URL을 버킷 이름과 키로 분리.
    """
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc.split('.')[0]
    key = parsed_url.path.lstrip('/')
    return bucket_name, key

def fetch_image_from_s3(bucket_name: str, image_key: str) -> bytes:
    """
    S3에서 이미지를 가져와 바이트로 반환.
    """
    # 환경 변수 확인
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION")

    if not all([aws_access_key, aws_secret_key, aws_region]):
        raise ValueError("AWS credentials or region are missing in environment variables.")

    # S3 클라이언트 생성
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region,
    )

    # 이미지 가져오기
    response = s3.get_object(Bucket=bucket_name, Key=image_key)
    return response['Body'].read()
