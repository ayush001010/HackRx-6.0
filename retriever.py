from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from config import EMBEDDING_MODEL
from document_manager import DocumentManager
from pdf_loader import PDFLoader

class VectorStoreProvider:
    def __init__(self, manager: DocumentManager):
        self.manager = manager
        self.retriever = self._create_retriever()

    def _create_retriever(self) -> VectorStoreRetriever:
        loader = PDFLoader(self.manager.get_filepath())
        raw_documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(raw_documents)

        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma.from_documents(split_docs, embeddings)
        return vector_store.as_retriever(search_kwargs={'k': 8})

    def get_retriever(self) -> VectorStoreRetriever:
        return self.retriever