"""
Builder for assembling context for LLM prompts
"""
from typing import List, Dict, Any, Optional, Union
import re

from app.document_processing.vector_store import VectorStore
from app.database.snowflake_connector import SnowflakeConnector
from app.database.schema_utils import SchemaUtils

class ContextBuilder:
    """Assembles context for LLM prompts from different sources"""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        snowflake_connector: Optional[SnowflakeConnector] = None
    ):
        """
        Initialize context builder
        
        Args:
            vector_store: Vector store for document retrieval
            snowflake_connector: Snowflake connector for database info
        """
        self.vector_store = vector_store
        self.snowflake_connector = snowflake_connector
        
        if snowflake_connector:
            self.schema_utils = SchemaUtils(snowflake_connector)
        else:
            self.schema_utils = None
    
    def build_context(
        self,
        query: str,
        include_docs: bool = True,
        include_db_schema: bool = True,
        doc_limit: int = 5
    ) -> str:
        """
        Build context for a query from all available sources
        
        Args:
            query: User query
            include_docs: Whether to include document context
            include_db_schema: Whether to include database schema
            doc_limit: Maximum number of documents to include
            
        Returns:
            Assembled context string
        """
        context_parts = []
        
        # Add document context if available and requested
        if include_docs and self.vector_store:
            doc_context = self.get_document_context(query, limit=doc_limit)
            if doc_context:
                context_parts.append("# DOCUMENT CONTEXT")
                context_parts.append(doc_context)
        
        # Add database schema if available and requested
        if include_db_schema and self.schema_utils:
            db_context = self.get_database_context(query)
            if db_context:
                context_parts.append("# DATABASE SCHEMA")
                context_parts.append(db_context)
        
        # Combine all context parts
        if context_parts:
            return "\n\n".join(context_parts)
        else:
            return ""
    
    def get_document_context(self, query: str, limit: int = 5) -> str:
        """
        Get relevant document context for a query
        
        Args:
            query: User query
            limit: Maximum number of documents to include
            
        Returns:
            Document context string
        """
        if not self.vector_store:
            return ""
            
        relevant_docs = self.vector_store.search_documents(
            query=query,
            top_k=limit
        )
        
        if not relevant_docs:
            return ""
            
        # Format documents
        doc_texts = []
        for i, doc in enumerate(relevant_docs):
            # Extract source information
            source = doc.metadata.get("source", "Unknown")
            
            # Format the document text
            doc_text = f"Document {i+1} (Source: {source}):\n{doc.page_content}"
            doc_texts.append(doc_text)
            
        return "\n\n".join(doc_texts)
    
    def get_database_context(self, query: str) -> str:
        """
        Get relevant database schema context for a query
        
        Args:
            query: User query
            
        Returns:
            Database schema context string
        """
        if not self.schema_utils:
            return ""
            
        # For simple implementation, just return the full schema
        # In a more advanced implementation, we could filter tables based on the query
        return self.schema_utils.get_schema_context(include_samples=False)
    
    def is_sql_query(self, query: str) -> bool:
        """
        Determine if a query is likely to require SQL generation
        
        Args:
            query: User query
            
        Returns:
            True if query likely needs SQL, False otherwise
        """
        # Simple heuristic for SQL detection
        sql_indicators = [
            r"\bselect\b", r"\bfrom\b", r"\bwhere\b", r"\bgroup by\b",
            r"\border by\b", r"\bjoin\b", r"\binsert\b", r"\bupdate\b",
            r"\bdelete\b", r"\bcreate\b", r"\balter\b", r"\bdrop\b",
            r"database", r"table", r"query", r"sql", r"snowflake"
        ]
        
        query_lower = query.lower()
        for indicator in sql_indicators:
            if re.search(indicator, query_lower):
                return True
                
        return False
