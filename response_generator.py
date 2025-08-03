# Backend/Bajaj/response_generator.py

import os
import requests
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from models import QueryResponse, GeneratedQueries
from retriever import VectorStoreProvider
from config import GOOGLE_API_KEY # Ensure you have this import

class ResponseGenerator:
    """
    Handles the generation of responses for a set of questions using a RAG approach.
    """
    def __init__(self, vector_store_provider: VectorStoreProvider):
        """
        Initializes the generator with a vector store provider.
        """
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not configured.")
            
        self.vector_store_provider = vector_store_provider
        
        # Initialize the large language model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0.0
        )

        # Define the prompt for the LLM
        self.prompt = PromptTemplate(
            template="""You are a helpful assistant. Answer the user's question only based on the provided context.
            If the answer cannot be found in the context, say "I am sorry, but the provided text does not contain
            information about [topic of question]." Do not make up any information.
            
            Context: {context}
            Question: {input}
            
            Provide the answer and a rationale for your answer.
            """,
            input_variables=["context", "input"]
        )

    def _generate_response_for_question(self, question: str) -> Tuple[QueryResponse, GeneratedQueries]:
        """
        Generates a single response for a given question.
        """
        # Create a document combining chain
        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        # Create the retrieval chain
        # This will use the retriever from our VectorStoreProvider
        retrieval_chain = create_retrieval_chain(
            self.vector_store_provider.retriever, # Accessing the retriever directly
            document_chain
        )
        
        # Invoke the chain to get the response
        response = retrieval_chain.invoke({"input": question})
        
        # Extract the answer, rationale, and source page from the response
        answer_text = response.get("answer", "No answer found.").strip()

        # The model's response might contain both the answer and rationale
        # We need to parse this. A simple approach is to look for key phrases.
        rationale_text = "Rationale not explicitly provided by the model."
        if "Rationale:" in answer_text:
            parts = answer_text.split("Rationale:")
            answer_text = parts[0].strip()
            rationale_text = parts[1].strip()

        # Extract the source pages from the retrieved documents
        source_pages = []
        if 'context' in response:
            for doc in response['context']:
                source_pages.append(doc.metadata.get('source', 'Unknown'))
        
        return QueryResponse(
            answer=answer_text,
            rationale=rationale_text,
            source_page=list(set(source_pages)) # Use set to get unique pages
        ), GeneratedQueries(generated_queries=[]) # Generated queries are not implemented here for simplicity

    def generate_response(self, question: str) -> Tuple[QueryResponse, GeneratedQueries]:
        """
        Generates a response for a single question.
        """
        return self._generate_response_for_question(question)
