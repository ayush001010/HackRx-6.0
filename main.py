# --- Compatibility Fix for Python >=3.10 ---
import collections
import collections.abc
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence
    
# —————— Azure Blob Storage Setup ——————
from dotenv import load_dotenv
load_dotenv()

from azure.storage.blob import BlobServiceClient
import os
import tempfile

account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
container_name = os.getenv("AZURE_STORAGE_CONTAINER")
account_key = os.getenv("AZURE_STORAGE_KEY")

if not account_url or not container_name or not account_key:
    raise RuntimeError("Missing Azure Blob Storage env vars")

blob_service = BlobServiceClient(account_url=account_url, credential=account_key)
container_client = blob_service.get_container_client(container_name)

def upload_blob(blob_name: str, data: bytes):
    blob_client = container_client.get_blob_client(blob=blob_name)
    blob_client.upload_blob(data, overwrite=True)

def download_blob_to_path(blob_name: str, local_path: str):
    blob_client = container_client.get_blob_client(blob=blob_name)
    with open(local_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

# ————————————————————————————————
import time
import traceback
from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
from rich import print as rprint
from rich.panel import Panel
import requests
from config import *
from models import QueryRequest, QueryResponse, FinalAnswer, Question
from query_service import QueryService

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Batch-Optimized RAG System",
    description="Processes documents and answers a list of questions together with high efficiency.",
    version="1.0.0",
)

query_service = QueryService()
security = HTTPBearer()

# --- Security Dependency ---
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# --- Application Events ---
@app.on_event("startup")
def on_startup():
    if not GOOGLE_API_KEY:
        raise RuntimeError("Missing critical environment variable: GOOGLE_API_KEY")
    rprint(Panel("Application startup complete. API token loaded.", title="[green]System Status[/green]"))

# --- Middleware ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f} sec"
    rprint(f"[cyan]Request[/cyan] '{request.method} {request.url.path}' [bold green]completed in {process_time:.4f}s[/bold green]")
    return response

# --- API Endpoints ---
@app.post(
    "/api/v1/hackrx/run",
    response_model=QueryResponse,
    tags=["Query Processing"],
    summary="Process a Document and Answer a Batch of Questions",
    status_code=status.HTTP_200_OK
)
async def run_submission(
    request_body: QueryRequest,
    authenticated: bool = Depends(verify_token)
):
    rprint(Panel(f"Processing request for document: [blue]{str(request_body.documents)}[/blue]", title="[cyan]New Request[/cyan]"))
    try:
        # 1. Convert question strings into Question models
        questions_as_models = [Question(question=q) for q in request_body.questions]

        # 2. Handle Azure Blob PDFs (if provided as blob://filename.pdf)
        document_url = str(request_body.documents)
        if document_url.startswith("blob://"):
            blob_name = document_url.replace("blob://", "")
            local_pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
            download_blob_to_path(blob_name, local_pdf_path)
            document_url = local_pdf_path

        # 3. Run your existing query processing logic
        results: List[FinalAnswer] = query_service.process_queries(
            document_url=document_url,
            questions=questions_as_models,
        )

        final_answers = [result.answer for result in results]
        return QueryResponse(answers=final_answers)

    except requests.exceptions.RequestException as e:
        rprint(Panel(f"[bold red]Document Download Failed:[/bold red]\n{e}", title="[red]Error[/red]"))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to download document: {e}")
    except ValueError as e:
        rprint(Panel(f"[bold red]Processing Error:[/bold red]\n{e}", title="[red]Error[/red]"))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        tb_str = traceback.format_exc()
        rprint(Panel(f"[bold red]An unexpected server error occurred:[/bold red]\n{tb_str}", title="[red]Server Error[/red]"))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred.")

@app.get("/health", tags=["Monitoring"], summary="API Health Check")
def health_check():
    return {"status": "ok"}
