import base64
import mimetypes
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.vectorstores import VectorStoreRetriever
from models import AnswerWithRationale
from config import LLM_MODEL,GOOGLE_API_KEY

def image_to_base64_uri(file_path: str):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:{mime_type};base64,{encoded_string}"

class GraphState(TypedDict):
    question: str
    retriever: VectorStoreRetriever
    documents: List[Document]
    generation: AnswerWithRationale

class RAGWorkflow:
    def __init__(self):
        self.graph = self._build_graph()

    def _retrieval_node(self, state: GraphState):
        documents = state["retriever"].invoke(state["question"])
        return {"documents": documents}

    def _generation_node(self, state: GraphState):
        text_context = "\n\n---\n\n".join([doc.page_content for doc in state["documents"]])
        
        prompt_text = (
            "You are an expert Q&A system. Based ONLY on the following text context and any accompanying images, "
            "provide a clear answer and a rationale for your answer.\n\n"
            "TEXT CONTEXT:\n{context}\n\n"
            "QUESTION: {question}"
        ).format(context=text_context, question=state["question"])

        message_parts = [{"type": "text", "text": prompt_text}]
        
        image_paths = []
        for doc in state["documents"]:
            image_paths.extend(doc.metadata.get("image_paths", []))
        
        if image_paths:
            unique_image_paths = sorted(list(set(image_paths)))
            for image_path in unique_image_paths:
                image_uri = image_to_base64_uri(image_path)
                message_parts.append({"type": "image_url", "image_url": {"url": image_uri}})

        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GOOGLE_API_KEY, temperature=0)
        structured_llm = llm.with_structured_output(AnswerWithRationale)
        generation = structured_llm.invoke([HumanMessage(content=message_parts)])
        return {"generation": generation}

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self._retrieval_node)
        workflow.add_node("generate", self._generation_node)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def invoke(self, question: str, retriever: VectorStoreRetriever):
        initial_state = {"question": question, "retriever": retriever}
        return self.graph.invoke(initial_state)