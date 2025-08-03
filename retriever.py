# Backend/Bajaj/retriever.py

import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_manager import DocumentManager
from pdf_loader import PDFLoader
from docx_loader import DocxLoader
from email_loader import EmailLoader
from config import EMBEDDING_MODEL

class VectorStoreProvider:
    def __init__(self, manager: DocumentManager):
        self.manager = manager
        self.retriever = self._create_retriever()

    def _create_retriever(self) -> VectorStoreRetriever:
        file_path = self.manager.get_filepath()
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            loader = PDFLoader(file_path)
        elif file_extension == ".docx":
            loader = DocxLoader(file_path)
        elif file_extension in [".eml", ".msg"]:
            loader = EmailLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        print(f"Loading document with {type(loader).__name__}...")
        raw_documents = loader.load()

        if not raw_documents:
            raise ValueError(f"Could not load any content from {file_path}")

        # Split documents into smaller, more manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        split_docs = text_splitter.split_documents(raw_documents)

        # Create embeddings for the document chunks
        embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Create an in-memory vector store with Chroma
        db = Chroma.from_documents(split_docs, embedding=embedding_model)

        # Return a retriever that fetches the top 4 most relevant chunks
        return db.as_retriever(search_kwargs={"k": 4})