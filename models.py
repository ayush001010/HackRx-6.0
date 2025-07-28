from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answer: str
    rationale: str
    source_image_urls: Optional[List[str]] = None

class AnswerWithRationale(BaseModel):
    answer: str
    rationale: str