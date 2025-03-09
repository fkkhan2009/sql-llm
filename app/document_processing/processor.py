"""
Document processing utilities for chunking and preprocessing
"""
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import settings

class DocumentProcessor:
    """Handles document chunking and preprocessing"""
    
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP
    ):
        """
        Initialize the document processor
        
        Args:
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process a list of documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of processed Document chunks
        """
        return self.chunk_documents(documents)
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Document chunks
        """
        return self.text_splitter.split_documents(documents)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Basic cleaning operations
        text = text.strip()
        # Remove extra whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """
        Extract and enhance metadata from a document
        
        Args:
            document: Document object
            
        Returns:
            Enhanced metadata dictionary
        """
        metadata = document.metadata.copy()
        
        # Add document length
        metadata['length'] = len(document.page_content)
        
        # Extract document snippet
        max_snippet_length = 100
        content = document.page_content
        if len(content) > max_snippet_length:
            metadata['snippet'] = content[:max_snippet_length] + '...'
        else:
            metadata['snippet'] = content
            
        return metadata
