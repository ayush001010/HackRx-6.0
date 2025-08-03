import os
from langchain_core.documents import Document
from email.parser import BytesParser
from email.policy import default
import extract_msg

class EmailLoader:
    """
    A loader to extract content from email files (.eml and .msg).
    It extracts the subject, sender, and body of the email.
    """

    def __init__(self, file_path: str):
        """
        Initializes the loader with the path to the email file.
        
        Args:
            file_path: The full path to the email file.
        """
        self.file_path = file_path
        self.file_extension = os.path.splitext(self.file_path)[1].lower()

    def load(self) -> list[Document]:
        """
        Loads the email file, extracts relevant information (sender, subject, body),
        and returns a single LangChain Document object.
        
        Returns:
            A list containing one Document object.
        """
        if self.file_extension == ".eml":
            return self._load_eml()
        elif self.file_extension == ".msg":
            return self._load_msg()
        else:
            print(f"Unsupported email file type: {self.file_path}")
            return []

    def _load_eml(self) -> list[Document]:
        """Loads and parses a .eml file."""
        try:
            with open(self.file_path, 'rb') as fp:
                message = BytesParser(policy=default).parse(fp)

            subject = message.get("Subject", "")
            sender = message.get("From", "")
            body = ""

            # Check if the message has a plain text part
            if message.is_multipart():
                for part in message.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(part.get_content_charset(), errors='ignore')
                        break
            else:
                if message.get_content_type() == "text/plain":
                    body = message.get_payload(decode=True).decode(message.get_content_charset(), errors='ignore')

            content = f"Subject: {subject}\nFrom: {sender}\n\n{body}"
            metadata = {"source": os.path.basename(self.file_path), "subject": subject, "sender": sender}
            return [Document(page_content=content, metadata=metadata)]

        except Exception as e:
            print(f"Error loading EML file {self.file_path}: {e}")
            return []

    def _load_msg(self) -> list[Document]:
        """Loads and parses a .msg file using extract-msg."""
        try:
            with extract_msg.Message(self.file_path) as msg:
                subject = msg.subject
                sender = msg.sender
                body = msg.body

                content = f"Subject: {subject}\nFrom: {sender}\n\n{body}"
                metadata = {"source": os.path.basename(self.file_path), "subject": subject, "sender": sender}
                return [Document(page_content=content, metadata=metadata)]

        except Exception as e:
            print(f"Error loading MSG file {self.file_path}: {e}")
            return []
