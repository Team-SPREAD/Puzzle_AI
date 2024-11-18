import os
import boto3
from io import BytesIO
from urllib.parse import urlparse

def parse_s3_url(s3_url: str):
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc.split('.')[0]
    key = parsed_url.path.lstrip('/')
    return bucket_name, key

def fetch_image_from_s3(bucket_name: str, image_key: str) -> bytes:
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )
    response = s3.get_object(Bucket=bucket_name, Key=image_key)
    return response['Body'].read()
