import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.0-flash"
IMAGE_DIR = "static/extracted_images"
IMAGE_ENDPOINT = "/extracted_images"

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY must be set in the .env file.")