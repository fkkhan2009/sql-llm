"""
Templates for LLM prompts
"""

class PromptTemplates:
    """Collection of prompt templates for different tasks"""
    
    @staticmethod
    def qa_prompt(query: str, context: str = "") -> str:
        """
        Generate a prompt for question answering
        
        Args:
            query: User query
            context: Context information
            
        Returns:
            Formatted prompt
        """
        if context:
            return f"""You are a helpful assistant that answers questions based on the provided context. If the answer cannot be found in the context, say that you don't know.

Context information:
{context}

Question: {query}

Answer:"""
        else:
            return f"""You are a helpful assistant that answers questions based on your knowledge. If you don't know the answer, please say so.

Question: {query}

Answer:"""
    
    @staticmethod
    def sql_generation_prompt(query: str, context: str = "") -> str:
        """
        Generate a prompt for SQL query generation
        
        Args:
            query: User query
            context: Context information (database schema)
            
        Returns:
            Formatted prompt
        """
        return f"""You are a helpful assistant that generates Snowflake SQL queries based on user requests. Always follow these guidelines:
1. Include appropriate column names and table references based on the schema.
2. Use standard Snowflake SQL syntax.
3. Add comments to explain complex parts of the query.
4. If you're unclear about the schema, make reasonable assumptions and note them.

Database schema information:
{context}

User request: {query}

Generate a Snowflake SQL query that fulfills this request:"""
    
    @staticmethod
    def system_prompt() -> str:
        """
        Generate a system prompt for the assistant
        
        Returns:
            System prompt string
        """
        return """You are a helpful AI assistant with access to documents and database information. Your responses are based on the provided context and your knowledge. When answering questions:

1. If you are presented with document context, prioritize that information.
2. If you are presented with database schema information, use it to understand the data structure.
3. If asked to generate SQL, make sure the query is valid for Snowflake and uses the correct table and column names.
4. Always be honest if you don't know the answer or if the information is not in the provided context.
5. Keep your responses concise and focused on the user's question."""
    
    @staticmethod
    def combined_prompt(query: str, context: str = "", is_sql_query: bool = False) -> str:
        """
        Generate a combined prompt based on query type
        
        Args:
            query: User query
            context: Context information
            is_sql_query: Whether this is likely an SQL query
            
        Returns:
            Formatted prompt
        """
        if is_sql_query:
            return PromptTemplates.sql_generation_prompt(query, context)
        else:
            return PromptTemplates.qa_prompt(query, context)
