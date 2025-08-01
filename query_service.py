from typing import List
from fastapi import BackgroundTasks
from document_manager import DocumentManager
from retriever import VectorStoreProvider
from workflow import RAGWorkflow
from models import QueryResponse

class QueryService:
    def __init__(self):
        self.rag_workflow = RAGWorkflow()

    def process_queries(
        self,
        document_url: str,
        questions: List[str],
        background_tasks: BackgroundTasks
    ):
        manager = DocumentManager(document_url)
        vector_store_provider = VectorStoreProvider(manager)
        retriever = vector_store_provider.get_retriever()
        
        full_results = []
        for question in questions:
            final_state, generated_queries = self.rag_workflow.invoke(question, retriever)
            
            generation = final_state["generation"]
            retrieved_docs = final_state["documents"]

            source_page = None
            if retrieved_docs and 'page' in retrieved_docs[0].metadata:
                source_page = retrieved_docs[0].metadata['page']

            query_response = QueryResponse(
                answer=generation.answer,
                rationale=generation.rationale,
                source_page=source_page
            )
            
            full_results.append((query_response, generated_queries))

        background_tasks.add_task(manager.cleanup)
        return full_results