from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from models import AnswerWithRationale, GeneratedQueries
from typing import Optional
from config import ANSWER_LLM_MODEL,QUERY_LLM_MODEL,GOOGLE_API_KEY

class GraphState(TypedDict):
    original_question: str
    decomposed_questions: List[str]
    retriever: VectorStoreRetriever
    documents: List[Document]
    generation: AnswerWithRationale

class RAGWorkflow:
    def __init__(self):
        self.generation_llm = ChatGoogleGenerativeAI(model=ANSWER_LLM_MODEL, api_key=GOOGLE_API_KEY, temperature=0)
        self.decomposition_llm = ChatGoogleGenerativeAI(model=QUERY_LLM_MODEL, api_key=GOOGLE_API_KEY, temperature=0)
        self.graph = self._build_graph()

    def _query_decomposition_node(self, state: GraphState):
        prompt = ChatPromptTemplate.from_template(
            "You are an expert research assistant tasked with query understanding and rewriting.\n"
            "Your job is to generate 3 diverse, relevant, and high-quality search queries based on the original user question.\n\n"
            "Your queries should:\n"
            "- Target the core intent of the original question.\n"
            "- Vary slightly in phrasing to explore related angles or subtopics.\n"
            "- Be suitable for use in a document retrieval system.\n\n"
            "NOTE: Output must be a valid Pydantic object of type `GeneratedQueries`, with a `queries` field containing exactly 3 strings.\n\n"
            "Think deeply before answering.\n"
            "Original Question: {question}"
        )
        structured_llm = self.decomposition_llm.with_structured_output(GeneratedQueries)
        chain = prompt | structured_llm
        generated_queries: Optional[GeneratedQueries] = chain.invoke({"question": state["original_question"]})
        if not generated_queries:
            return {"decomposed_questions": [state["original_question"]]}
        decomposed_questions = generated_queries.queries
        decomposed_questions.append(state["original_question"])
        return {"decomposed_questions": decomposed_questions}

    def _retrieval_node(self, state: GraphState):
        all_retrieved_docs = []
        for q in state["decomposed_questions"]:
            retrieved = state["retriever"].invoke(q)
            all_retrieved_docs.extend(retrieved)
        unique_docs_dict = {doc.page_content: doc for doc in all_retrieved_docs}
        return {"documents": list(unique_docs_dict.values())}

    def _generation_node(self, state: GraphState):
        context = "\n\n---\n\n".join([doc.page_content for doc in state["documents"]])
        prompt = ChatPromptTemplate.from_template(
            "You are a highly knowledgeable assistant answering questions using the given context ONLY.\n"
            "Provide a concise, accurate answer and a clear rationale strictly based on the provided content.\n\n"
            "CONTEXT:\n{context}\n\n"
            "QUESTION: {question}\n\n"
            "INSTRUCTIONS:\n"
            "- Use only the information in the context to formulate the answer.\n"
            "- Avoid making assumptions or using external knowledge.\n"
            "- Your response should be a valid Pydantic object of type `AnswerWithRationale` with two fields:\n"
            "  - `answer`: A direct, fact-based response.\n"
            "  - `rationale`: A short explanation justifying the answer using information from the context."
        )
        structured_llm = self.generation_llm.with_structured_output(AnswerWithRationale)
        chain = prompt | structured_llm
        generation = chain.invoke({
            "context": context,
            "question": state["original_question"]
        })
        return {"generation": generation}

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("decompose_query", self._query_decomposition_node)
        workflow.add_node("retrieve", self._retrieval_node)
        workflow.add_node("generate", self._generation_node)
        workflow.set_entry_point("decompose_query")
        workflow.add_edge("decompose_query", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def invoke(self, question: str, retriever: VectorStoreRetriever):
        initial_state = {"original_question": question, "retriever": retriever}
        final_state = self.graph.invoke(initial_state)
        decomposed_questions = final_state.get("decomposed_questions", [])
        return final_state, decomposed_questions