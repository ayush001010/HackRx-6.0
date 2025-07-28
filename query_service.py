import os
from typing import List
from fastapi import BackgroundTasks
from document_manager import DocumentManager
from retriever import VectorStoreRetriever
from workflow import RAGWorkflow
from models import QueryResponse
from config import IMAGE_ENDPOINT

class QueryService:
    def __init__(self):
        self.rag_workflow = RAGWorkflow()

    def process_queries(
        self,
        document_url: str,
        questions: List[str],
        background_tasks: BackgroundTasks
    ) -> List[QueryResponse]:
        
        manager = DocumentManager(document_url)
        vector_store_provider = VectorStoreRetriever(manager)
        retriever = vector_store_provider.get_retriever()
        
        responses = []
        for question in questions:
            final_state = self.rag_workflow.invoke(question, retriever)
            generation = final_state["generation"]
            retrieved_docs = final_state["documents"]

            all_image_paths = []
            if retrieved_docs:
                for doc in retrieved_docs:
                    all_image_paths.extend(doc.metadata.get("image_paths", []))
            
            unique_image_urls = sorted(list(set(
                f"{IMAGE_ENDPOINT}/{os.path.basename(p)}" for p in all_image_paths
            )))

            responses.append(
                QueryResponse(
                    answer=generation.answer,
                    rationale=generation.rationale,
                    source_image_urls=unique_image_urls or None
                )
            )

        background_tasks.add_task(manager.cleanup)
        return responses