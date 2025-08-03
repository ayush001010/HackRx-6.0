import docx2txt
import os
from langchain_core.documents import Document

class DocxLoader:
    """
    A simple loader to extract text from a .docx file.
    It uses the docx2txt library to convert the document to plain text.
    """

    def __init__(self, file_path: str):
        """
        Initializes the loader with the path to the DOCX file.
        
        Args:
            file_path: The full path to the DOCX file.
        """
        self.file_path = file_path

    def load(self) -> list[Document]:
        """
        Loads the DOCX file, extracts all text, and returns a single
        LangChain Document object.
        
        Returns:
            A list containing one Document object.
        """
        try:
            # docx2txt.process returns the full text content as a string
            content = docx2txt.process(self.file_path)
            
            # Create and return a single Document object with the content
            # and a minimal set of metadata.
            metadata = {"source": os.path.basename(self.file_path)}
            return [Document(page_content=content, metadata=metadata)]

        except Exception as e:
            # Handle potential errors during file processing
            print(f"Error loading DOCX file {self.file_path}: {e}")
            return []

