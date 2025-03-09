"""
Orchestrator for LLM interactions
"""
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from app.llm.client import LLMClient
from app.llm.context_builder import ContextBuilder
from app.llm.prompt_templates import PromptTemplates
from app.document_processing.vector_store import VectorStore
from app.database.snowflake_connector import SnowflakeConnector

class LLMOrchestrator:
    """Manages LLM interactions and response generation"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        context_builder: Optional[ContextBuilder] = None,
        vector_store: Optional[VectorStore] = None,
        snowflake_connector: Optional[SnowflakeConnector] = None
    ):
        """
        Initialize LLM orchestrator
        
        Args:
            llm_client: Ollama client for LLM interactions
            context_builder: Context builder for assembling context
            vector_store: Vector store for document retrieval
            snowflake_connector: Snowflake connector for database interactions
        """
        self.llm_client = llm_client
        
        # Create context builder if not provided
        if context_builder:
            self.context_builder = context_builder
        else:
            self.context_builder = ContextBuilder(
                vector_store=vector_store,
                snowflake_connector=snowflake_connector
            )
        
        self.vector_store = vector_store
        self.snowflake_connector = snowflake_connector
    
    def generate_response(
        self,
        query: str,
        include_docs: bool = True,
        include_db_schema: bool = True,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate a response to a user query
        
        Args:
            query: User query
            include_docs: Whether to include document context
            include_db_schema: Whether to include database schema
            temperature: Sampling temperature
            
        Returns:
            Response object with generated text and metadata
        """
        # Determine if this is likely an SQL query
        is_sql_query = self.context_builder.is_sql_query(query)
        
        # Build context
        context = self.context_builder.build_context(
            query=query,
            include_docs=include_docs,
            include_db_schema=include_db_schema if is_sql_query else False
        )
        
        # Generate prompt
        prompt = PromptTemplates.combined_prompt(
            query=query,
            context=context,
            is_sql_query=is_sql_query
        )
        
        # Get system prompt
        system_prompt = PromptTemplates.system_prompt()
        
        # Generate response from LLM
        response_text = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        # If this was an SQL query, try to execute it
        sql_result = None
        if is_sql_query and self.snowflake_connector:
            sql_query = self.extract_sql_query(response_text)
            if sql_query:
                success, result = self.snowflake_connector.execute_query(sql_query)
                if success:
                    # Convert DataFrame to serializable format
                    if isinstance(result, pd.DataFrame):
                        # Convert to dict with records orientation for proper serialization
                        sql_result = {
                            "columns": result.columns.tolist(),
                            "data": result.to_dict(orient='records'),
                            "shape": result.shape
                        }
                    else:
                        # For non-DataFrame results (like affected rows message)
                        sql_result = {"message": str(result)}
        
        # Create response object
        response = {
            "query": query,
            "response": response_text,
            "is_sql_query": is_sql_query,
            "sql_result": sql_result
        }
        
        return response
    
    def extract_sql_query(self, text: str) -> Optional[str]:
        """
        Extract SQL query from LLM response
        
        Args:
            text: LLM response text
            
        Returns:
            Extracted SQL query or None if not found
        """
        # Try to find SQL query between triple backticks
        import re
        sql_matches = re.findall(r"```sql\n(.*?)```", text, re.DOTALL)
        
        if sql_matches:
            return sql_matches[0].strip()
            
        # Try without language specifier
        sql_matches = re.findall(r"```\n(.*?)```", text, re.DOTALL)
        if sql_matches:
            return sql_matches[0].strip()
            
        # Try to find the query based on common SQL keywords
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if any(line.strip().upper().startswith(keyword) for keyword in sql_keywords):
                # Extract from this line until the end or until we hit a natural boundary
                query_lines = []
                for j in range(i, len(lines)):
                    if lines[j].strip() and not lines[j].startswith(("Here", "This", "The", "Note")):
                        query_lines.append(lines[j])
                    else:
                        break
                return "\n".join(query_lines)
                
        return None
    
    def generate_sql(
        self,
        query: str,
        include_db_schema: bool = True,
        temperature: float = 0.3
    ) -> Tuple[str, Optional[str]]:
        """
        Generate SQL for a query
        
        Args:
            query: User query about data
            include_db_schema: Whether to include database schema
            temperature: Sampling temperature
            
        Returns:
            Tuple of (generated SQL, explanation)
        """
        # Build database context
        context = ""
        if include_db_schema and self.context_builder:
            context = self.context_builder.get_database_context(query)
        
        # Generate prompt
        prompt = PromptTemplates.sql_generation_prompt(
            query=query,
            context=context
        )
        
        # Get system prompt
        system_prompt = PromptTemplates.system_prompt()
        
        # Generate SQL from LLM
        response_text = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        # Extract SQL query
        sql_query = self.extract_sql_query(response_text)
        
        # Return both the SQL and the explanation
        return sql_query, response_text
