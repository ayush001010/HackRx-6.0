import os
import re
import uuid
import fitz
from langchain_core.documents import Document
from typing import List

class PDFLoader:
    def __init__(self, file_path: str, image_dir: str):
        self.file_path = file_path
        self.image_dir = image_dir
        os.makedirs(image_dir, exist_ok=True)

    def _is_table_line(self, line_text: str) -> bool:
        return bool(re.search(r"(\s{2,}|\t)", line_text)) and len(line_text.strip()) > 10

    def _extract_images(self, page, page_number: int) -> List[str]:
        image_paths = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(page.parent, xref)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image_filename = f"page_{page_number+1}_img_{img_index}_{uuid.uuid4().hex}.png"
            image_path = os.path.join(self.image_dir, image_filename)
            pix.save(image_path)
            image_paths.append(image_path)
            pix = None
        return image_paths

    def load(self) -> list[Document]:
        documents = []
        with fitz.open(self.file_path) as doc:
            for page_number in range(len(doc)):
                page = doc.load_page(page_number)
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
                image_paths = self._extract_images(page, page_number)
                
                combined_text = ""
                if full_text:
                    combined_text += "### Text Content ###\n" + full_text + "\n"
                if table_text:
                    combined_text += "\n### Table Content ###\n" + table_text + "\n"
                if image_paths:
                    image_basenames = [os.path.basename(p) for p in image_paths]
                    combined_text += "\n### Associated Images ###\n" + "\n".join(image_basenames) + "\n"

                documents.append(
                    Document(
                        page_content=combined_text.strip(),
                        metadata={
                            "page": page_number + 1,
                            "source_file": os.path.basename(self.file_path),
                            "image_paths": image_paths,
                        }
                    )
                )
        return documents