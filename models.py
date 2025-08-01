from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answer: str
    rationale: str
    source_page: Optional[int] = None

class AnswerWithRationale(BaseModel):
    answer: str
    rationale: str

class GeneratedQueries(BaseModel):
    queries: List[str] = Field(description="A list of 3 distinct, self-contained search queries based on the original question.")