from google.cloud import vision

def extract_text_with_google_vision(image_bytes: bytes) -> str:
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(f"Google Vision API Error: {response.error.message}")
    return response.full_text_annotation.text
