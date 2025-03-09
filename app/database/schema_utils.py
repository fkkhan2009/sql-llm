"""
Enhanced utilities for working with database schema information
"""
from typing import Dict, List, Any, Optional, Set
import json
from app.database.snowflake_connector import SnowflakeConnector

class SchemaUtils:
    """Utilities for working with database schema information"""
    
    def __init__(self, connector: SnowflakeConnector):
        """
        Initialize with a database connector
        
        Args:
            connector: SnowflakeConnector instance
        """
        self.connector = connector
    
    def get_schema_context(self, 
                           database: Optional[str] = None, 
                           schema: Optional[str] = None, 
                           include_samples: bool = False,
                           max_tables: int = 20) -> str:
        """
        Generate a comprehensive schema context for the LLM
        
        Args:
            database: Database name
            schema: Schema name
            include_samples: Whether to include sample data
            max_tables: Maximum number of tables to include
            
        Returns:
            Formatted schema context
        """
        db = database or self.connector.database
        sch = schema or self.connector.schema
        
        if not db or not sch:
            return "No database or schema specified."
        
        # Get table list
        tables = self.connector.get_tables(db, sch)
        if not tables:
            return f"No tables found in {db}.{sch}."
        
        # Limit the number of tables if needed
        if len(tables) > max_tables:
            tables = tables[:max_tables]
            truncated_message = f"\n(Showing {max_tables} tables out of {len(tables)} total tables)"
        else:
            truncated_message = ""
        
        # Build context
        context = [f"Database: {db}", f"Schema: {sch}", f"Tables: {', '.join(tables)}{truncated_message}\n"]
        
        # Add details for each table
        for table in tables:
            context.append(f"TABLE: {table}")
            columns = self.connector.get_table_schema(table, db, sch)
            
            # Format columns
            column_details = []
            for col in columns:
                nullable = "NULL" if col.get("nullable", "Y") == "Y" else "NOT NULL"
                column_details.append(f"  - {col['name']} ({col['type']}) {nullable}")
            
            context.append("Columns:")
            context.extend(column_details)
            
            # Add sample data if requested
            if include_samples:
                sample_data = self.connector.get_table_sample(table, limit=3, database=db, schema=sch)
                if not sample_data.empty:
                    context.append("\nSample data:")
                    context.append(sample_data.to_string(index=False))
            
            context.append("\n")  # Empty line between tables
        
        return "\n".join(context)
    
    def get_schema_context_for_tables(self,
                                      tables: List[str],
                                      database: Optional[str] = None,
                                      schema: Optional[str] = None,
                                      include_samples: bool = False) -> str:
        """
        Generate schema context for specific tables
        
        Args:
            tables: List of table names to include
            database: Database name
            schema: Schema name
            include_samples: Whether to include sample data
            
        Returns:
            Formatted schema context for the specified tables
        """
        db = database or self.connector.database
        sch = schema or self.connector.schema
        
        if not db or not sch:
            return "No database or schema specified."
        
        if not tables:
            return "No tables specified for schema context."
            
        # Verify tables exist
        all_tables = set(self.connector.get_tables(db, sch))
        valid_tables = [table for table in tables if table in all_tables]
        
        if not valid_tables:
            return f"None of the specified tables exist in {db}.{sch}."
            
        # Build context
        context = [f"Database: {db}", f"Schema: {sch}", f"Selected Tables: {', '.join(valid_tables)}\n"]
        
        # Add details for each table
        for table in valid_tables:
            context.append(f"TABLE: {table}")
            columns = self.connector.get_table_schema(table, db, sch)
            
            # Format columns
            column_details = []
            for col in columns:
                nullable = "NULL" if col.get("nullable", "Y") == "Y" else "NOT NULL"
                column_details.append(f"  - {col['name']} ({col['type']}) {nullable}")
            
            context.append("Columns:")
            context.extend(column_details)
            
            # Add sample data if requested
            if include_samples:
                sample_data = self.connector.get_table_sample(table, limit=3, database=db, schema=sch)
                if not sample_data.empty:
                    context.append("\nSample data:")
                    context.append(sample_data.to_string(index=False))
            
            context.append("\n")  # Empty line between tables
            
        # Add relationship information for the tables
        context.append("POTENTIAL RELATIONSHIPS:")
        relationships = self._identify_potential_relationships(valid_tables, db, sch)
        if relationships:
            for rel in relationships:
                context.append(f"  - {rel}")
        else:
            context.append("  No clear relationships identified between the selected tables.")
            
        return "\n".join(context)
    
    def _identify_potential_relationships(self,
                                         tables: List[str],
                                         database: Optional[str] = None,
                                         schema: Optional[str] = None) -> List[str]:
        """
        Identify potential relationships between tables based on column names
        
        Args:
            tables: List of table names
            database: Database name
            schema: Schema name
            
        Returns:
            List of relationship descriptions
        """
        db = database or self.connector.database
        sch = schema or self.connector.schema
        
        if not tables or len(tables) < 2:
            return []
            
        # Get schemas for all tables
        table_schemas = {}
        for table in tables:
            table_schemas[table] = self.connector.get_table_schema(table, db, sch)
            
        relationships = []
        
        # Look for potential foreign keys
        for table1 in tables:
            for table2 in tables:
                if table1 == table2:
                    continue
                    
                # Check for columns in table1 that reference table2
                for col in table_schemas[table1]:
                    col_name = col["name"].lower()
                    table2_name = table2.lower()
                    
                    # Common foreign key patterns
                    if (col_name == f"{table2_name}_id" or 
                        col_name == table2_name + "id" or
                        col_name == "id_" + table2_name or
                        col_name == f"fk_{table2_name}" or
                        (table2_name.endswith('s') and col_name == f"{table2_name[:-1]}_id")):
                        relationships.append(f"{table1}.{col['name']} likely references {table2}")
                        
                    # Check for exact column name matches (especially 'id')
                    if col_name == "id":
                        for col2 in table_schemas[table2]:
                            if col2["name"].lower() == f"{table1.lower()}_id":
                                relationships.append(f"{table2}.{col2['name']} likely references {table1}.id")
        
        return relationships
    
    def generate_sql_hints(self, database: Optional[str] = None, schema: Optional[str] = None) -> str:
        """
        Generate SQL hints for Snowflake based on schema
        
        Args:
            database: Database name
            schema: Schema name
            
        Returns:
            SQL hints as a formatted string
        """
        db = database or self.connector.database
        sch = schema or self.connector.schema
        
        if not db or not sch:
            return "No database or schema information available for SQL hints."
        
        hints = [
            "# Snowflake SQL Hints",
            f"# For database: {db}, schema: {sch}\n",
            "1. Use fully qualified names (DATABASE.SCHEMA.TABLE) for clarity",
            "2. Filter with WHERE clauses before performing JOIN operations when possible",
            "3. Use CASE statements for conditional logic",
            "4. For date/time filtering, use date functions like DATE_TRUNC()",
            "5. Leverage Snowflake-specific functions like ARRAY_AGG() and OBJECT_AGG() for aggregation",
            "6. Use LIMIT for testing queries on large tables",
            "7. Remember that Snowflake is case-insensitive by default",
            "8. For performance, reference specific columns instead of SELECT *"
        ]
        
        return "\n".join(hints)
    
    def get_table_relationships(self, database: Optional[str] = None, schema: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
        """
        Attempt to determine relationships between tables based on column names
        This is a heuristic approach (not using actual foreign key constraints)
        
        Args:
            database: Database name
            schema: Schema name
            
        Returns:
            Dictionary mapping tables to their potential relationships
        """
        db = database or self.connector.database
        sch = schema or self.connector.schema
        
        if not db or not sch:
            return {}
        
        tables = self.connector.get_tables(db, sch)
        table_schemas = {}
        relationships = {}
        
        # Get schema for all tables
        for table in tables:
            table_schemas[table] = self.connector.get_table_schema(table, db, sch)
        
        # Look for potential relationships based on column names
        for table_name, columns in table_schemas.items():
            relationships[table_name] = []
            
            for column in columns:
                col_name = column["name"].lower()
                
                # Look for columns that might be foreign keys
                if col_name.endswith("_id") or col_name == "id" or "_fk" in col_name:
                    # Find potential related tables
                    potential_table = col_name.replace("_id", "").replace("_fk", "")
                    
                    # Check if the potential table exists
                    for other_table in tables:
                        if other_table.lower() == potential_table or other_table.lower() + "s" == potential_table:
                            relationships[table_name].append({
                                "column": column["name"],
                                "potential_reference": other_table,
                                "confidence": "high" if other_table.lower() == potential_table else "medium"
                            })
        
        return relationships
    
    def suggest_join_conditions(self, table1: str, table2: str) -> List[str]:
        """
        Suggest potential join conditions between two tables
        
        Args:
            table1: First table name
            table2: Second table name
            
        Returns:
            List of potential join conditions
        """
        db = self.connector.database
        sch = self.connector.schema
        
        if not db or not sch:
            return []
        
        # Get schemas
        schema1 = self.connector.get_table_schema(table1, db, sch)
        schema2 = self.connector.get_table_schema(table2, db, sch)
        
        if not schema1 or not schema2:
            return []
        
        suggestions = []
        
        # Get column names
        columns1 = [col["name"].lower() for col in schema1]
        columns2 = [col["name"].lower() for col in schema2]
        
        # Check for exact match columns
        for col1 in schema1:
            col1_name = col1["name"].lower()
            for col2 in schema2:
                col2_name = col2["name"].lower()
                
                # If column names match exactly
                if col1_name == col2_name:
                    suggestions.append(f"{table1}.{col1['name']} = {table2}.{col2['name']}")
                
                # Check for foreign key patterns
                if col1_name == f"{table2.lower()}_id":
                    suggestions.append(f"{table1}.{col1['name']} = {table2}.id")
                
                if col2_name == f"{table1.lower()}_id":
                    suggestions.append(f"{table1}.id = {table2}.{col2['name']}")
        
        return suggestions
    
    def get_complex_schema_context(self, 
                                 database: Optional[str] = None, 
                                 schema: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive schema context as a structured dictionary
        
        Args:
            database: Database name
            schema: Schema name
            
        Returns:
            Dictionary with complete schema information
        """
        db = database or self.connector.database
        sch = schema or self.connector.schema
        
        if not db or not sch:
            return {"error": "No database or schema specified."}
        
        result = {
            "database": db,
            "schema": sch,
            "tables": {},
            "relationships": {}
        }
        
        # Get all tables
        tables = self.connector.get_tables(db, sch)
        
        # Get details for each table
        for table in tables:
            columns = self.connector.get_table_schema(table, db, sch)
            
            # Get a small sample of data
            sample = self.connector.get_table_sample(table, limit=2, database=db, schema=sch)
            sample_data = []
            if not sample.empty:
                for _, row in sample.iterrows():
                    sample_data.append(dict(row))
            
            result["tables"][table] = {
                "columns": columns,
                "sample_data": sample_data
            }
        
        # Get potential relationships
        result["relationships"] = self.get_table_relationships(db, sch)
        
        return result
    
    def format_schema_for_context(self, database: Optional[str] = None, schema: Optional[str] = None) -> str:
        """
        Format schema information in a way that's optimized for LLM context
        
        Args:
            database: Database name
            schema: Schema name
            
        Returns:
            Formatted schema information string
        """
        db = database or self.connector.database
        sch = schema or self.connector.schema
        
        context = self.get_schema_context(db, sch, include_samples=False)
        hints = self.generate_sql_hints(db, sch)
        
        return f"{context}\n\n{hints}"
