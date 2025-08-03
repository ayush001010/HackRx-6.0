import os
import uuid
import requests

class DocumentManager:
    def __init__(self, document_url: str):
        self.document_url = document_url
        self.file_path = self._download_document()
        self.DIR = 'doc_cache'
        os.makedirs(self.DIR, exist_ok=True)


    def _download_document(self) -> str:
        response = requests.get(str(self.document_url), timeout=60)
        response.raise_for_status()
        unique_id = uuid.uuid4()
        file_path = os.path.join(self.DIR, f"{unique_id}.pdf")
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path

    def get_filepath(self) -> str:
        return self.file_path

    def cleanup(self):
        if self.file_path and os.path.exists(self.file_path):
            os.remove(self.file_path)