# Backend/Bajaj/main.py

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import ValidationError
from typing import List
import os
import requests # <-- ADDED THIS IMPORT for exception handling
from dotenv import load_dotenv

# Import your existing modules
from models import QueryRequest, QueryResponse
from query_service import QueryService
from config import GOOGLE_API_KEY

# Load environment variables
load_dotenv()

# --- Configuration for Authentication ---
# The API_AUTH_TOKEN will be loaded from your .env file
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")

# FastAPI's security scheme for Bearer token
security = HTTPBearer()

# --- Dependency for Authentication ---
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verifies the Bearer token provided in the Authorization header.
    """
    if credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Must be Bearer."
        )
    if credentials.credentials != API_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token."
        )
    return True

# --- FastAPI Application Setup ---
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="API for processing large documents and making contextual decisions.",
    version="1.0.0",
)

# Initialize your QueryService
query_service = QueryService()

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=List[QueryResponse], status_code=status.HTTP_200_OK)
async def run_submission(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_token)
):
    """
    Processes a document URL and a list of questions to provide contextual answers.
    """
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: GOOGLE_API_KEY is not set."
        )
    if not API_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: API_AUTH_TOKEN is not set."
        )

    try:
        results_with_generated_queries = query_service.process_queries(
            document_url=request.documents,
            questions=request.questions,
            background_tasks=background_tasks
        )

        # Extract only the QueryResponse objects for the final API response
        final_responses: List[QueryResponse] = [res for res, _ in results_with_generated_queries]

        return final_responses

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download document from URL: {e}"
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid request data: {e.errors()}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {e}"
        )
