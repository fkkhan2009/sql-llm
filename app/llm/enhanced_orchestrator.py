"""
Enhanced orchestrator for LLM interactions with multi-step reasoning
"""
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import time
import re
import logging
from datetime import datetime
import json

from app.llm.client import LLMClient
from app.database.snowflake_connector import SnowflakeConnector
from app.database.schema_utils import SchemaUtils
from app.document_processing.vector_store import VectorStore
from app.llm.context_builder import ContextBuilder

# Set up logging
logger = logging.getLogger(__name__)

class DatabaseResponse:
    """Represents a complete response to a database query"""

    def __init__(
        self,
        question: str,
        sql: str,
        results: Any,
        explanation: str,
        relevant_tables: List[str] = None,
        view_metadata: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None
    ):
        """
        Initialize a database response
        
        Args:
            question: Original user question
            sql: Generated SQL query
            results: Query execution results
            explanation: Human-friendly explanation
            relevant_tables: Tables identified as relevant to the query
            view_metadata: Metadata about created view, if any
            execution_time: Time taken to execute query, if available
        """
        self.question = question
        self.sql = sql
        self.results = results
        self.explanation = explanation
        self.relevant_tables = relevant_tables or []
        self.view_metadata = view_metadata
        self.execution_time = execution_time
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result_data = None
        if isinstance(self.results, pd.DataFrame):
            result_data = {
                "columns": self.results.columns.tolist(),
                "data": self.results.to_dict(orient='records'),
                "row_count": len(self.results)
            }
        else:
            result_data = {"message": str(self.results)}
            
        return {
            "question": self.question,
            "sql": self.sql,
            "results": result_data,
            "explanation": self.explanation,
            "relevant_tables": self.relevant_tables,
            "view_metadata": self.view_metadata,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp
        }

class EnhancedLLMOrchestrator:
    """Manages multi-step LLM interactions for database queries"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        snowflake_connector: SnowflakeConnector,
        schema_utils: SchemaUtils,
        context_builder: Optional[ContextBuilder] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        """
        Initialize enhanced orchestrator
        
        Args:
            llm_client: LLM client for model interactions
            snowflake_connector: Snowflake connector for database interactions
            schema_utils: Schema utilities for database schema handling
            context_builder: Context builder for assembling context
            vector_store: Vector store for document retrieval
        """
        self.llm_client = llm_client
        self.snowflake_connector = snowflake_connector
        self.schema_utils = schema_utils
        self.context_builder = context_builder
        self.vector_store = vector_store
    
    async def process_database_query(
        self,
        question: str,
        workspace_id: Optional[str] = None,
        include_docs: bool = True,
        save_as_view: bool = False,
        view_name: Optional[str] = None,
        view_description: Optional[str] = None,
        temperature: float = 0.3,
        use_progressive_building: bool = True
    ) -> DatabaseResponse:
        """
        Process a database question with multi-step LLM reasoning
        
        Args:
            question: User's natural language question
            workspace_id: Optional workspace for context
            include_docs: Whether to include document context
            save_as_view: Whether to save results as a view
            view_name: Name for the view if saving
            view_description: Description for the view if saving
            temperature: Temperature for LLM generation
            use_progressive_building: Whether to use progressive query building
            
        Returns:
            DatabaseResponse with complete query results
        """
        # Record start time for performance tracking
        start_time = time.time()
        
        # Step 1: Get document context if available and requested
        doc_context = ""
        if include_docs and workspace_id and self.context_builder:
            doc_context = self.context_builder.get_document_context(question)
        
        # Step 2: Analyze and identify relevant tables
        relevant_tables = await self._identify_relevant_tables(question, doc_context, temperature)
        
        # Step 3: Get focused schema information for the relevant tables
        focused_schema = self.schema_utils.get_schema_context_for_tables(relevant_tables)
        
        # Step 4: Generate SQL with focused context - with progressive building or direct generation
        if use_progressive_building:
            sql_query = await self._build_query_progressively(
                question,
                relevant_tables,
                focused_schema,
                temperature
            )
        else:
            sql_query = await self._generate_sql_query(
                question, 
                focused_schema, 
                doc_context,
                temperature
            )
            
            # Validate and optimize the SQL query
            sql_query = await self._validate_and_optimize_sql(sql_query, temperature)
        
        # Step 5: Execute the SQL query with error handling
        success = False
        error_message = None
        results = None
        attempt = 0
        max_attempts = 3  # Maximum refinement attempts
        
        while not success and attempt < max_attempts:
            attempt += 1
            
            try:
                success, results = self.snowflake_connector.execute_query(sql_query)
                
                if not success:
                    error_message = str(results) if hasattr(results, '__str__') else "Unknown error"
                    logger.warning(f"Query attempt {attempt} failed: {error_message}")
                    
                    # Attempt to fix the query
                    sql_query = await self._handle_query_error(
                        sql_query, 
                        error_message, 
                        focused_schema,
                        attempt,
                        max_attempts
                    )
            except Exception as e:
                error_message = str(e)
                logger.warning(f"Query attempt {attempt} error: {error_message}")
                
                # Attempt to fix the query
                sql_query = await self._handle_query_error(
                    sql_query, 
                    error_message,
                    focused_schema,
                    attempt,
                    max_attempts
                )
        
        # If still not successful after all attempts, provide a useful error message
        if not success:
            results = f"Query execution failed after {max_attempts} attempts. Last error: {error_message}"
        
        # Step 6: Create view if requested
        view_metadata = None
        if success and save_as_view and view_name:
            view_metadata = await self._create_view_from_query(
                sql_query, 
                view_name, 
                view_description, 
                workspace_id,
                temperature
            )
        
        # Step 7: Generate explanation of results
        explanation = await self._explain_results(
            question, 
            sql_query, 
            results, 
            view_metadata,
            temperature
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Build and return complete response
        return DatabaseResponse(
            question=question,
            sql=sql_query,
            results=results,
            explanation=explanation,
            relevant_tables=relevant_tables,
            view_metadata=view_metadata,
            execution_time=execution_time
        )
    
    async def _identify_relevant_tables(
        self, 
        question: str, 
        doc_context: str = "",
        temperature: float = 0.3
    ) -> List[str]:
        """
        Identify tables relevant to the user's question
        
        Args:
            question: User's natural language question
            doc_context: Document context if available
            temperature: Temperature for LLM generation
            
        Returns:
            List of table names
        """
        # Get all available tables
        all_tables = self.snowflake_connector.get_tables()
        tables_info = "\n".join([f"- {table}" for table in all_tables])
        
        prompt = f"""You are a database expert. Given the user's question, identify which database tables are most likely relevant.
        
User Question: {question}

{f"Additional Context: {doc_context}" if doc_context else ""}

Available Tables:
{tables_info}

First, think through what information would be needed to answer this question.
Then identify which tables from the available list would contain this information.

Return only the names of the relevant tables as a JSON array, like this:
["table_name1", "table_name2"]

Relevant Tables:"""
        
        response = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature
        )
        
        # Extract JSON array from response
        try:
            # Find anything that looks like a JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                tables_json = json_match.group(0)
                tables = json.loads(tables_json)
                return tables
            
            # Fallback: try to parse the whole response
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error extracting tables from response: {str(e)}")
            # Fallback: extract table names using regex
            table_matches = re.findall(r'["\']([\w\d_]+)["\']', response)
            return list(set(table_matches))
    
    async def _generate_sql_query(
        self, 
        question: str, 
        schema_context: str,
        doc_context: str = "",
        temperature: float = 0.3
    ) -> str:
        """
        Generate SQL query for the user's question
        
        Args:
            question: User's natural language question
            schema_context: Database schema context
            doc_context: Document context if available
            temperature: Temperature for LLM generation
            
        Returns:
            SQL query string
        """
        prompt = f"""You are a database expert. Generate a Snowflake SQL query to answer the user's question.

User Question: {question}

{f"Additional Context: {doc_context}" if doc_context else ""}

Database Schema Information:
{schema_context}

Important Guidelines:
1. Use proper Snowflake SQL syntax.
2. Include clear column names with table aliases.
3. Use appropriate joins and conditions.
4. Add comments to explain complex parts.
5. Format the query for readability.
6. Make sure column references in WHERE, GROUP BY, or ORDER BY clauses use the correct table aliases.
7. If calculations or aggregations are needed, name them clearly.

Return only the SQL query, without any additional text or explanation.

SQL Query:"""
        
        sql_query = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature
        )
        
        # Extract SQL from response (if it contains markdown code blocks)
        sql_match = re.search(r'```sql\n(.*?)\n```', sql_query, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Try without language specifier
        sql_match = re.search(r'```\n(.*?)\n```', sql_query, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
            
        return sql_query.strip()
    
    async def _validate_and_optimize_sql(
        self, 
        sql_query: str,
        temperature: float = 0.2
    ) -> str:
        """
        Validate and optimize the generated SQL query
        
        Args:
            sql_query: Generated SQL query
            temperature: Temperature for LLM generation
            
        Returns:
            Validated and optimized SQL query
        """
        prompt = f"""You are a Snowflake SQL expert. Review and optimize the following SQL query:

```sql
{sql_query}
```

Your task:
1. Check for syntax errors and fix them
2. Ensure all table aliases are used consistently
3. Add appropriate JOIN conditions if missing
4. Optimize the query for better performance
5. Make sure column references are unambiguous
6. Add appropriate comments to explain the logic

Return only the optimized SQL query, without any additional text or explanation.

Optimized SQL Query:"""
        
        optimized_sql = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature
        )
        
        # Extract SQL from response
        sql_match = re.search(r'```sql\n(.*?)\n```', optimized_sql, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Try without language specifier
        sql_match = re.search(r'```\n(.*?)\n```', optimized_sql, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
            
        return optimized_sql.strip()
    
    async def _create_view_from_query(
        self,
        sql_query: str,
        view_name: str,
        view_description: Optional[str] = None,
        workspace_id: Optional[str] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Create a database view from the query
        
        Args:
            sql_query: SQL query to use for the view
            view_name: Name for the view
            view_description: Description for the view
            workspace_id: Workspace ID for metadata
            temperature: Temperature for LLM generation
            
        Returns:
            View metadata
        """
        # Prepare view name - ensure it's safe and prefixed with workspace
        safe_view_name = view_name.replace(" ", "_").upper()
        if workspace_id:
            prefixed_view_name = f"WS_{workspace_id[:8]}_{safe_view_name}"
        else:
            prefixed_view_name = f"VIEW_{safe_view_name}"
            
        # Generate CREATE VIEW statement
        prompt = f"""Generate a CREATE OR REPLACE VIEW statement for Snowflake using this query:

```sql
{sql_query}
```

The view should be named: {prefixed_view_name}
{f"Description: {view_description}" if view_description else ""}

Make sure to:
1. Include appropriate column names in the view definition
2. Add a comment with the description
3. Use proper Snowflake syntax

Return only the CREATE VIEW statement without any other text.

CREATE VIEW statement:"""
        
        create_view_sql = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature
        )
        
        # Extract SQL from response
        sql_match = re.search(r'```sql\n(.*?)\n```', create_view_sql, re.DOTALL)
        if sql_match:
            create_view_sql = sql_match.group(1).strip()
        else:
            # Try without language specifier
            sql_match = re.search(r'```\n(.*?)\n```', create_view_sql, re.DOTALL)
            if sql_match:
                create_view_sql = sql_match.group(1).strip()
        
        # Execute the CREATE VIEW statement
        success, result = self.snowflake_connector.execute_query(create_view_sql)
        
        # Create metadata record
        view_metadata = {
            "name": prefixed_view_name,
            "description": view_description or "Generated view",
            "original_name": view_name,
            "workspace_id": workspace_id,
            "created_at": datetime.now().isoformat(),
            "query": sql_query,
            "success": success
        }
        
        if not success:
            view_metadata["error"] = str(result)
            
        return view_metadata
    
    async def _explain_results(
        self,
        question: str,
        sql_query: str,
        results: Any,
        view_metadata: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a human-friendly explanation of the results
        
        Args:
            question: Original user question
            sql_query: SQL query that was executed
            results: Query execution results
            view_metadata: Metadata about created view, if any
            temperature: Temperature for LLM generation
            
        Returns:
            Natural language explanation of results
        """
        # Format results for the prompt
        if isinstance(results, pd.DataFrame):
            # Format the first few rows as a table
            max_rows = min(5, len(results))
            results_text = results.head(max_rows).to_string(index=False)
            row_count = len(results)
            
            if max_rows < row_count:
                results_text += f"\n[{row_count - max_rows} more rows not shown]"
        else:
            # For non-DataFrame results
            results_text = str(results)
        
        view_text = ""
        if view_metadata and view_metadata.get("success", False):
            view_text = f"\nResults have been saved to a view named '{view_metadata.get('name')}'."
            
        prompt = f"""You are a helpful data analyst. Explain the following SQL query results in a clear, concise way.

Original Question: {question}

SQL Query Used:
```sql
{sql_query}
```

Query Results:
```
{results_text}
```
{view_text}

Provide:
1. A summary of what the query did
2. Key insights from the results
3. Any patterns or trends in the data
4. A direct answer to the original question
5. Potential follow-up questions the user might have

Your explanation:"""
        
        explanation = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature
        )
        
        return explanation
    
    async def _handle_query_error(
        self, 
        query: str, 
        error_message: str, 
        schema_context: str,
        attempt: int = 1,
        max_attempts: int = 3
    ) -> str:
        """
        Handle SQL execution errors by refining the query
        
        Args:
            query: Failed SQL query
            error_message: Error returned by Snowflake
            schema_context: Database schema context
            attempt: Current attempt number
            max_attempts: Maximum number of refinement attempts
            
        Returns:
            Refined SQL query
        """
        if attempt > max_attempts:
            # Return a simplified query rather than giving up entirely
            return self._generate_fallback_query(query, error_message, schema_context)
        
        # Create prompt for error correction
        prompt = f"""You are an expert SQL developer. The following Snowflake SQL query failed:
        
```sql
{query}
```

The error message was:
```
{error_message}
```

Database schema information:
{schema_context}

Please fix the query to resolve this specific error. Explain your fix briefly, then provide ONLY the corrected SQL query.
"""
        
        response = self.llm_client.generate(prompt=prompt, temperature=0.3)
        
        # Extract the corrected SQL
        import re
        sql_match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Try without language specifier
        sql_match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Extract anything that looks like SQL if no code blocks found
        for line in response.split("\n"):
            if "SELECT" in line.upper() or "WITH" in line.upper():
                # Found likely SQL statement, extract until end of text or next code block
                start_idx = response.index(line)
                end_match = re.search(r'```', response[start_idx:])
                if end_match:
                    return response[start_idx:start_idx + end_match.start()].strip()
                else:
                    return response[start_idx:].strip()
        
        # Fallback to the parts of the response after explanation
        return response.split("\n\n")[-1].strip()
        
    def _generate_fallback_query(self, query: str, error_message: str, schema_context: str) -> str:
        """
        Generate a simplified fallback query when multiple refinement attempts fail
        
        Args:
            query: Failed SQL query
            error_message: Error returned by Snowflake
            schema_context: Database schema context
            
        Returns:
            Simplified SQL query
        """
        # Extract table names from the original query
        table_pattern = r'FROM\s+([^\s,;()]+)|JOIN\s+([^\s,;()]+)'
        table_matches = re.findall(table_pattern, query, re.IGNORECASE)
        
        # Flatten and clean table names
        tables = []
        for match in table_matches:
            for table in match:
                if table and table.strip():
                    tables.append(table.strip())
        
        # If we found tables, generate a simple query
        if tables:
            primary_table = tables[0]
            return f"SELECT * FROM {primary_table} LIMIT 10"
        else:
            # Extreme fallback - just get sample data from any table mentioned in schema
            schema_tables = re.findall(r'TABLE:\s+([^\s]+)', schema_context)
            if schema_tables:
                return f"SELECT * FROM {schema_tables[0]} LIMIT 10"
            
        # Ultimate fallback
        return "SELECT 1 AS fallback_column"
    
    async def _generate_select_clause(
        self,
        question: str,
        tables: List[str],
        schema_context: str,
        temperature: float = 0.3
    ) -> str:
        """
        Generate the SELECT clause for a query
        
        Args:
            question: User's question
            tables: Relevant tables
            schema_context: Database schema context
            temperature: Temperature for LLM generation
            
        Returns:
            SELECT clause
        """
        prompt = f"""Generate ONLY the SELECT clause for a Snowflake SQL query to answer this question:

Question: {question}

Available tables: {', '.join(tables)}

Database schema:
{schema_context}

Your task is to identify which columns should be selected to answer the question.
Include appropriate aggregate functions, CASE statements, or calculations if needed.
Include aliases for clarity where appropriate.

Return ONLY the SELECT clause (starting with SELECT and ending before FROM), nothing else:"""
        
        response = self.llm_client.generate(prompt=prompt, temperature=temperature)
        
        # Clean up response
        if "SELECT" not in response.upper():
            response = "SELECT " + response
        
        # Make sure it's just the SELECT clause
        select_clause = response.strip()
        
        # Remove any FROM clause if accidentally included
        from_position = select_clause.upper().find(" FROM ")
        if from_position > 0:
            select_clause = select_clause[:from_position].strip()
        
        return select_clause

    async def _generate_from_clause(
        self,
        question: str,
        tables: List[str],
        schema_context: str,
        temperature: float = 0.3
    ) -> str:
        """
        Generate the FROM and JOIN clauses for a query
        
        Args:
            question: User's question
            tables: Relevant tables
            schema_context: Database schema context
            temperature: Temperature for LLM generation
            
        Returns:
            FROM and JOIN clauses
        """
        prompt = f"""Generate ONLY the FROM and JOIN clauses for a Snowflake SQL query to answer this question:

Question: {question}

Available tables: {', '.join(tables)}

Database schema:
{schema_context}

Your task is to determine:
1. Which table should be the main table in the FROM clause
2. Which JOINs are needed to connect relevant tables
3. The appropriate join conditions

Return ONLY the FROM and JOIN clauses (starting with FROM and ending before WHERE/GROUP BY/ORDER BY), nothing else:"""
        
        response = self.llm_client.generate(prompt=prompt, temperature=temperature)
        
        # Clean up response
        if "FROM" not in response.upper():
            response = "FROM " + response
        
        # Make sure it only contains FROM and JOIN parts
        from_clause = response.strip()
        
        # Remove WHERE, GROUP BY, etc. if added
        for clause in ["WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT"]:
            position = from_clause.upper().find(f" {clause} ")
            if position > 0:
                from_clause = from_clause[:position].strip()
        
        return from_clause

    async def _generate_where_clause(
        self,
        question: str,
        tables: List[str],
        schema_context: str,
        temperature: float = 0.3
    ) -> str:
        """
        Generate the WHERE clause for a query
        
        Args:
            question: User's question
            tables: Relevant tables
            schema_context: Database schema context
            temperature: Temperature for LLM generation
            
        Returns:
            WHERE clause or empty string if not needed
        """
        prompt = f"""Generate ONLY the WHERE clause for a Snowflake SQL query to answer this question:

Question: {question}

Available tables: {', '.join(tables)}

Database schema:
{schema_context}

Your task is to identify what filtering conditions are needed to answer the question.
Include date ranges, categories, or other conditions implied by the question.
Make sure column references use proper table aliases if needed.

Return ONLY the WHERE clause (starting with WHERE and ending before GROUP BY/ORDER BY), 
or return "NO WHERE CLAUSE NEEDED" if filtering is not required:"""
        
        response = self.llm_client.generate(prompt=prompt, temperature=temperature)
        
        if "NO WHERE CLAUSE NEEDED" in response.upper():
            return ""
        
        # Clean up response
        if "WHERE" not in response.upper():
            response = "WHERE " + response
        
        # Make sure it only contains WHERE part
        where_clause = response.strip()
        
        # Remove other clauses if added
        for clause in ["GROUP BY", "ORDER BY", "HAVING", "LIMIT"]:
            position = where_clause.upper().find(f" {clause} ")
            if position > 0:
                where_clause = where_clause[:position].strip()
        
        return where_clause

    async def _generate_group_order_clause(
        self,
        question: str,
        tables: List[str],
        schema_context: str,
        temperature: float = 0.3
    ) -> str:
        """
        Generate GROUP BY, HAVING, and ORDER BY clauses
        
        Args:
            question: User's question
            tables: Relevant tables
            schema_context: Database schema context
            temperature: Temperature for LLM generation
            
        Returns:
            Trailing clauses for the query
        """
        prompt = f"""Generate ONLY the GROUP BY, HAVING, and ORDER BY clauses (if needed) for a Snowflake SQL query to answer this question:

Question: {question}

Available tables: {', '.join(tables)}

Database schema:
{schema_context}

Based on the question:
1. Determine if grouping is needed (include GROUP BY)
2. Determine if filtering on aggregates is needed (include HAVING)
3. Determine how results should be sorted (include ORDER BY)
4. Consider if a result limit is appropriate (include LIMIT)

Return ONLY these clauses as needed, or return "NO ADDITIONAL CLAUSES NEEDED" if none are required:"""
        
        response = self.llm_client.generate(prompt=prompt, temperature=temperature)
        
        if "NO ADDITIONAL CLAUSES NEEDED" in response.upper():
            return ""
        
        # Clean up response
        additional_clauses = response.strip()
        
        # Check for and add missing clause keywords
        if "GROUP BY" in question.upper() and "GROUP BY" not in additional_clauses.upper():
            for line in additional_clauses.split("\n"):
                if "GROUP BY" not in line.upper() and any(word in line.upper() for word in ["GROUP", "GROUPING"]):
                    additional_clauses = additional_clauses.replace(line, "GROUP BY " + line.lstrip("Group by ").lstrip("group by "))
        
        if "ORDER BY" in question.upper() and "ORDER BY" not in additional_clauses.upper():
            for line in additional_clauses.split("\n"):
                if "ORDER BY" not in line.upper() and any(word in line.upper() for word in ["ORDER", "SORT"]):
                    additional_clauses = additional_clauses.replace(line, "ORDER BY " + line.lstrip("Order by ").lstrip("order by "))
        
        return additional_clauses
        
    async def _build_query_progressively(
        self,
        question: str,
        relevant_tables: List[str],
        schema_context: str,
        temperature: float = 0.3
    ) -> str:
        """
        Build a SQL query progressively, validating each part
        
        Args:
            question: User's question
            relevant_tables: List of relevant tables
            schema_context: Database schema context
            temperature: Temperature for generation
            
        Returns:
            Complete SQL query
        """
        logger.info(f"Building query progressively for question: {question}")
        logger.info(f"Relevant tables: {', '.join(relevant_tables)}")
        
        # Step 1: Generate SELECT clause
        select_clause = await self._generate_select_clause(
            question, 
            relevant_tables, 
            schema_context,
            temperature
        )
        logger.info(f"Generated SELECT clause: {select_clause}")
        
        # Step 2: Generate FROM clause with JOINs
        from_clause = await self._generate_from_clause(
            question, 
            relevant_tables, 
            schema_context,
            temperature
        )
        logger.info(f"Generated FROM clause: {from_clause}")
        
        # Step 3: Generate WHERE clause
        where_clause = await self._generate_where_clause(
            question, 
            relevant_tables, 
            schema_context,
            temperature
        )
        logger.info(f"Generated WHERE clause: {where_clause}")
        
        # Step 4: Generate GROUP BY, ORDER BY, etc.
        additional_clauses = await self._generate_group_order_clause(
            question, 
            relevant_tables, 
            schema_context,
            temperature
        )
        logger.info(f"Generated additional clauses: {additional_clauses}")
        
        # Step 5: Combine all parts
        query_parts = [select_clause, from_clause]
        
        if where_clause:
            query_parts.append(where_clause)
        
        if additional_clauses:
            query_parts.append(additional_clauses)
        
        # Combine into complete query
        sql_query = " ".join(query_parts)
        logger.info(f"Combined query: {sql_query}")
        
        # Step 6: Validate the complete query
        validated_query = await self._validate_and_optimize_sql(sql_query, temperature)
        logger.info(f"Validated query: {validated_query}")
        
        return validated_query