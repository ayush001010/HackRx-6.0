from pydantic import BaseModel, HttpUrl, Field
from typing import List

class Question(BaseModel):
    question: str

class FinalAnswer(BaseModel):
    answer: str

class GeneratedQueriesForEachQuestion(BaseModel):
    queries: List[str] = Field(description="A list of 3 distinct, self-contained search queries based on the original question.")

class GeneratedQueries(BaseModel):
    lst: List[GeneratedQueriesForEachQuestion] = Field(description="This is a list consisting of another set of nested lists which contain the generated queries for each question.")

class QueryResponse(BaseModel):
    answers: List[str]

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]