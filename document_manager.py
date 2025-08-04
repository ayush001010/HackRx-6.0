import os
import uuid
import requests
from typing import Tuple
import shelve

class DocumentManager:
    def __init__(self, document_url: str):
        self.document_url = document_url
        self.DIR = 'doc_cache'
        os.makedirs(self.DIR, exist_ok=True)

        # Try to fetch a cached path; if none, download and cache it
        cached_path = self._get_cached_path()
        if cached_path:
            # Cache hit
            self.file_path = cached_path
            self.filename = os.path.basename(cached_path)
        else:
            # Cache miss
            self.file_path, self.filename = self._download_and_cache()

    def _get_cached_path(self) -> str:
        """Return the cached file path for this URL, or '' if not present."""
        with shelve.open(os.path.join(self.DIR, 'cache')) as cache:
            return cache.get(self.document_url, '')

    def _download_and_cache(self) -> Tuple[str, str]:
        """Download the document, save it, cache the path, and return (path, name)."""
        print("Downloading file...")
        response = requests.get(self.document_url, timeout=60)
        response.raise_for_status()

        filename = f"{uuid.uuid4()}.pdf"
        file_path = os.path.join(self.DIR, filename)

        with open(file_path, "wb") as f:
            f.write(response.content)
        print('File downloaded.')
        # Store in cache
        with shelve.open(os.path.join(self.DIR, 'cache')) as cache:
            cache[self.document_url] = file_path

        return file_path, filename

    def get_filepath(self) -> str:
        return self.file_path

    def get_filename(self) -> str:
        return self.filename

    def cleanup(self):
        if getattr(self, 'file_path', None) and os.path.exists(self.file_path):
            os.remove(self.file_path)
