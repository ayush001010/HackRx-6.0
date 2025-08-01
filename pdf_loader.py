import os
import re
import fitz
from langchain_core.documents import Document
from typing import List

class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def _is_table_line(self, line_text: str) -> bool:
        return bool(re.search(r"(\s{2,}|\t)", line_text)) and len(line_text.strip()) > 10

    def load(self) -> list[Document]:
        documents = []
        with fitz.open(self.file_path) as doc:
            for page_number, page in enumerate(doc):
                page_dict = page.get_text("dict")
                blocks = page_dict.get("blocks", [])
                text_lines, table_lines = [], []
                for block in blocks:
                    if block["type"] == 0:
                        for line in block.get("lines", []):
                            line_text = " ".join([span["text"] for span in line.get("spans", [])]).strip()
                            if self._is_table_line(line_text):
                                table_lines.append(line_text)
                            else:
                                text_lines.append(line_text)
                
                full_text = "\n".join(text_lines).strip()
                table_text = "\n".join(table_lines).strip()
                
                combined_text = ""
                if full_text:
                    combined_text += "### Text Content ###\n" + full_text + "\n"
                if table_text:
                    combined_text += "\n### Table Content ###\n" + table_text + "\n"
                
                if combined_text:
                    documents.append(
                        Document(
                            page_content=combined_text.strip(),
                            metadata={
                                "page": page_number + 1,
                                "source_file": os.path.basename(self.file_path),
                            }
                        )
                    )
        return documents