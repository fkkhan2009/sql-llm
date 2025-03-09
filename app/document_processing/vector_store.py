"""
ChromaDB integration for vector storage and retrieval
"""
import os
from typing import List, Dict, Any, Optional, Union
import uuid
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

class VectorStore:
    """Manages interactions with ChromaDB for document storage and retrieval"""
    
    def __init__(
        self,
        persist_directory: str = settings.CHROMA_PERSIST_DIRECTORY,
        embedding_model_name: str = settings.EMBEDDING_MODEL,
        collection_name: str = "documents"
    ):
        """
        Initialize the vector store
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_model_name: Name of the embedding model to use
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create the persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize the embedding model
        # Check if we're using Ollama or HuggingFace
        if embedding_model_name.startswith("ollama:"):
            # Format should be "ollama:model_name"
            ollama_model = embedding_model_name.split(":")[1]
            self.embedding_model = OllamaEmbeddings(
                model=ollama_model,
                base_url=settings.OLLAMA_BASE_URL
            )
        else:
            # Default to HuggingFace
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        
        # Initialize ChromaDB with retry
        try:
            # First attempt with default settings
            chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False
                )
            )
        except ValueError as e:
            # If there's a conflict with existing database, open with allow_reset=True
            if "already exists" in str(e):
                print("Detected existing ChromaDB instance with different settings. Using existing settings.")
                chroma_client = chromadb.PersistentClient(
                    path=persist_directory
                )
        
        # Get or create the collection
        self._get_or_create_collection(chroma_client)
    
    def _get_or_create_collection(self, client: chromadb.Client) -> None:
        """
        Get or create the ChromaDB collection
        
        Args:
            client: ChromaDB client
        """
        # Check if collection exists, if not create it
        try:
            # Try to get existing collection first
            collections = client.list_collections()
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if collection_exists:
                # If collection exists, use it
                self.langchain_chroma = Chroma(
                    client=client,
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model
                )
            else:
                # Collection doesn't exist, create a new one
                self.langchain_chroma = Chroma.from_documents(
                    documents=[],  # Empty document list for initialization
                    embedding=self.embedding_model,
                    collection_name=self.collection_name,
                    client=client
                )
        except ValueError as e:
            print(f"ChromaDB error: {str(e)}")
            # Handle case where collection exists with different settings
            if "exists with different embedding function" in str(e):
                # Just use the existing collection with its current settings
                print(f"Using existing collection '{self.collection_name}' with its current settings")
                self.langchain_chroma = Chroma(
                    client=client,
                    collection_name=self.collection_name,
                    embedding_function=None  # Let it use the existing embedding
                )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of document IDs
        """
        # Generate IDs for documents if they don't have one
        document_ids = []
        for doc in documents:
            if 'id' not in doc.metadata:
                doc.metadata['id'] = str(uuid.uuid4())
            document_ids.append(doc.metadata['id'])
            
        # Add documents to ChromaDB
        self.langchain_chroma.add_documents(documents)
        self.langchain_chroma.persist()
        
        return document_ids
    
    def search_documents(
        self, 
        query: str, 
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for documents similar to the query
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter_dict: Optional metadata filter
            
        Returns:
            List of relevant Document objects
        """
        return self.langchain_chroma.similarity_search(
            query=query, 
            k=top_k,
            filter=filter_dict
        )
    
    def search_documents_with_scores(
        self, 
        query: str, 
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Search for documents with similarity scores
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter_dict: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples
        """
        return self.langchain_chroma.similarity_search_with_relevance_scores(
            query=query, 
            k=top_k,
            filter=filter_dict
        )
    
    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID
        
        Args:
            document_id: Document ID
            
        Returns:
            Document object or None if not found
        """
        results = self.langchain_chroma.get(
            where={"id": document_id}
        )
        
        if results and len(results) > 0:
            return results[0]
        return None
    
    def delete_document(self, document_id: str) -> None:
        """
        Delete a document from the vector store
        
        Args:
            document_id: Document ID
        """
        self.langchain_chroma.delete(
            ids=[document_id]
        )
        self.langchain_chroma.persist()
    
    def delete_collection(self) -> None:
        """Delete the entire collection"""
        self.langchain_chroma._collection.delete()
        self.langchain_chroma.persist()
