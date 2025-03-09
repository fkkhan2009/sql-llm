"""
Updated Snowflake connector with view support
"""
import os
import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import snowflake.connector
# from snowflake.connector.cursor import SnowflakeCursor
# from snowflake.connector.errors import ProgrammingError, DatabaseError

from app.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class SnowflakeConnector:
    """Manages connections and interactions with Snowflake"""
    
    def __init__(
        self,
        account: Optional[str] = settings.SNOWFLAKE_ACCOUNT,
        user: Optional[str] = settings.SNOWFLAKE_USER,
        password: Optional[str] = settings.SNOWFLAKE_PASSWORD,
        password_file: Optional[str] = settings.SNOWFLAKE_PASSWORD_FILE,
        private_key_file: Optional[str] = settings.SNOWFLAKE_PRIVATE_KEY_FILE,
        private_key_passphrase: Optional[str] = settings.SNOWFLAKE_PRIVATE_KEY_PASSPHRASE,
        database: Optional[str] = settings.SNOWFLAKE_DATABASE,
        schema: Optional[str] = settings.SNOWFLAKE_SCHEMA,
        warehouse: Optional[str] = settings.SNOWFLAKE_WAREHOUSE,
        role: Optional[str] = settings.SNOWFLAKE_ROLE,
    ):
        """
        Initialize Snowflake connector with connection parameters
        
        Args:
            account: Snowflake account identifier
            user: Snowflake username
            password: Snowflake password (lowest priority)
            password_file: Path to file containing password (medium priority)
            private_key_file: Path to private key file (highest priority)
            private_key_passphrase: Passphrase for private key if encrypted
            database: Default database
            schema: Default schema
            warehouse: Default warehouse
            role: User role
        """
        self.account = account
        self.user = user
        self.password = password
        self.password_file = password_file
        self.private_key_file = private_key_file
        self.private_key_passphrase = private_key_passphrase
        self.database = database
        self.schema = schema
        self.warehouse = warehouse
        self.role = role
        
        # Connection will be established when needed
        self.conn = None
    
    def connect(self) -> snowflake.connector.SnowflakeConnection:
        """
        Establish a connection to Snowflake using the most secure authentication method available
        
        Returns:
            Snowflake connection
            
        Raises:
            ValueError: If required connection parameters are missing
            ConnectionError: If connection to Snowflake fails
        """
        if not self.account or not self.user:
            raise ValueError("Snowflake account and user are required")
            
        try:
            conn_params = {
                "account": self.account,
                "user": self.user,
            }
            
            # Authentication priority:
            # 1. Private key authentication (most secure)
            # 2. Password from file
            # 3. Password from environment/settings (least secure)
            
            # Try private key authentication first
            if self.private_key_file and os.path.exists(self.private_key_file):
                try:
                    with open(self.private_key_file, "rb") as key_file:
                        p_key = key_file.read()
                    
                    from cryptography.hazmat.primitives import serialization
                    from cryptography.hazmat.backends import default_backend
                    
                    # Load the private key
                    if self.private_key_passphrase:
                        # If passphrase is provided, use it
                        p_key_obj = serialization.load_pem_private_key(
                            p_key,
                            password=self.private_key_passphrase.encode(),
                            backend=default_backend()
                        )
                    else:
                        # No passphrase
                        p_key_obj = serialization.load_pem_private_key(
                            p_key,
                            password=None,
                            backend=default_backend()
                        )
                    
                    # Generate the private key in the format snowflake connector expects
                    pkb = p_key_obj.private_bytes(
                        encoding=serialization.Encoding.DER,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    )
                    
                    # Add the private key to connection parameters
                    conn_params["private_key"] = pkb
                    logger.info("Using private key authentication for Snowflake")
                except Exception as e:
                    logger.error(f"Failed to load private key: {str(e)}")
                    # Fall back to other authentication methods
            
            # If private key authentication is not set up, try password from file
            if "private_key" not in conn_params and self.password_file and os.path.exists(self.password_file):
                try:
                    with open(self.password_file, "r") as pwd_file:
                        conn_params["password"] = pwd_file.read().strip()
                    logger.info("Using password from file for Snowflake authentication")
                except Exception as e:
                    logger.error(f"Failed to read password file: {str(e)}")
                    # Fall back to password from environment
            
            # If no other authentication method is available, use password from environment
            if "private_key" not in conn_params and "password" not in conn_params:
                if not self.password:
                    raise ValueError("No valid authentication method available for Snowflake")
                conn_params["password"] = self.password
                logger.info("Using password from environment for Snowflake authentication")
            
            # Add optional parameters if provided
            if self.database:
                conn_params["database"] = self.database
            if self.schema:
                conn_params["schema"] = self.schema
            if self.warehouse:
                conn_params["warehouse"] = self.warehouse
            if self.role:
                conn_params["role"] = self.role
            
            # Establish connection
            self.conn = snowflake.connector.connect(**conn_params)
            return self.conn
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            raise ConnectionError(f"Failed to connect to Snowflake: {str(e)}")
    
    def get_connection(self) -> snowflake.connector.SnowflakeConnection:
        """
        Get the current connection or establish a new one
        
        Returns:
            Snowflake connection
        """
        if self.conn is None or self.conn.is_closed():
            return self.connect()
        return self.conn
    
    def close_connection(self) -> None:
        """Close the Snowflake connection if open"""
        if self.conn and not self.conn.is_closed():
            self.conn.close()
            self.conn = None
    
    def execute_query(self, query: str) -> Tuple[bool, Union[pd.DataFrame, str]]:
        """
        Execute a SQL query and return the results
        
        Args:
            query: SQL query to execute
            
        Returns:
            Tuple of (success, result)
              - success: Boolean indicating if query executed successfully
              - result: DataFrame with results if successful, error message if not
        """
        start_time = time.time()
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            
            # Attempt to fetch results (for SELECT queries)
            try:
                data = cursor.fetchall()
                columns = [col[0] for col in cursor.description]
                df = pd.DataFrame(data, columns=columns)
                end_time = time.time()
                execution_time = end_time - start_time
                logger.info(f"Query executed in {execution_time:.2f} seconds, returned {len(df)} rows")
                return True, df
            except:
                # For non-SELECT queries (e.g., INSERT, UPDATE)
                affected_rows = cursor.rowcount
                end_time = time.time()
                execution_time = end_time - start_time
                logger.info(f"Non-SELECT query executed in {execution_time:.2f} seconds")
                return True, f"Query executed successfully. Rows affected: {affected_rows}"
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"Error executing query in {execution_time:.2f} seconds: {str(e)}")
            return False, f"Error executing query: {str(e)}"
        finally:
            # Don't close the connection, just the cursor
            if 'cursor' in locals() and cursor:
                cursor.close()
    
    def get_databases(self) -> List[str]:
        """
        Get list of available databases
        
        Returns:
            List of database names
        """
        success, result = self.execute_query("SHOW DATABASES")
        if success and isinstance(result, pd.DataFrame):
            return result["name"].tolist()
        return []
    
    def get_schemas(self, database: Optional[str] = None) -> List[str]:
        """
        Get list of schemas in the specified database
        
        Args:
            database: Database name (uses default if None)
            
        Returns:
            List of schema names
        """
        db = database or self.database
        if not db:
            return []
            
        query = f"SHOW SCHEMAS IN DATABASE {db}"
        success, result = self.execute_query(query)
        if success and isinstance(result, pd.DataFrame):
            return result["name"].tolist()
        return []
    
    def get_tables(self, database: Optional[str] = None, schema: Optional[str] = None) -> List[str]:
        """
        Get list of tables in the specified database and schema
        
        Args:
            database: Database name (uses default if None)
            schema: Schema name (uses default if None)
            
        Returns:
            List of table names
        """
        db = database or self.database
        sch = schema or self.schema
        if not db or not sch:
            return []
            
        query = f"SHOW TABLES IN {db}.{sch}"
        success, result = self.execute_query(query)
        if success and isinstance(result, pd.DataFrame):
            return result["name"].tolist()
        return []
    
    def get_views(self, database: Optional[str] = None, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of views in the specified database and schema
        
        Args:
            database: Database name (uses default if None)
            schema: Schema name (uses default if None)
            
        Returns:
            List of view objects with metadata
        """
        db = database or self.database
        sch = schema or self.schema
        if not db or not sch:
            return []
            
        query = f"SHOW VIEWS IN {db}.{sch}"
        success, result = self.execute_query(query)
        
        if not success or not isinstance(result, pd.DataFrame) or result.empty:
            return []
            
        views = []
        for _, row in result.iterrows():
            view_info = {
                "name": row["name"],
                "database": row["database_name"],
                "schema": row["schema_name"],
                "created_on": row.get("created_on", ""),
                "owner": row.get("owner", "")
            }
            
            # Try to get the view definition
            try:
                query = f"SELECT GET_DDL('VIEW', '{db}.{sch}.{row['name']}')"
                ddl_success, ddl_result = self.execute_query(query)
                if ddl_success and isinstance(ddl_result, pd.DataFrame) and not ddl_result.empty:
                    view_info["definition"] = ddl_result.iloc[0, 0]
            except Exception as e:
                logger.error(f"Error getting view definition: {str(e)}")
                view_info["definition"] = ""
                
            views.append(view_info)
            
        return views
    
    def get_view_info(self, view_name: str, database: Optional[str] = None, schema: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific view
        
        Args:
            view_name: View name
            database: Database name (uses default if None)
            schema: Schema name (uses default if None)
            
        Returns:
            View information dictionary or None if not found
        """
        db = database or self.database
        sch = schema or self.schema
        if not db or not sch:
            return None
            
        # Check if view exists
        query = f"SHOW VIEWS LIKE '{view_name}' IN {db}.{sch}"
        success, result = self.execute_query(query)
        
        if not success or not isinstance(result, pd.DataFrame) or result.empty:
            return None
            
        view_info = {
            "name": result.iloc[0]["name"],
            "database": result.iloc[0]["database_name"],
            "schema": result.iloc[0]["schema_name"],
            "created_on": result.iloc[0].get("created_on", ""),
            "owner": result.iloc[0].get("owner", "")
        }
        
        # Get view definition
        try:
            query = f"SELECT GET_DDL('VIEW', '{db}.{sch}.{view_name}')"
            ddl_success, ddl_result = self.execute_query(query)
            if ddl_success and isinstance(ddl_result, pd.DataFrame) and not ddl_result.empty:
                view_info["definition"] = ddl_result.iloc[0, 0]
                
                # Try to extract SQL from definition
                import re
                sql_match = re.search(r'AS\s+(.*)', view_info["definition"], re.IGNORECASE | re.DOTALL)
                if sql_match:
                    view_info["sql"] = sql_match.group(1).strip()
        except Exception as e:
            logger.error(f"Error getting view definition: {str(e)}")
            view_info["definition"] = ""
            
        # Get view columns
        try:
            query = f"DESCRIBE VIEW {db}.{sch}.{view_name}"
            desc_success, desc_result = self.execute_query(query)
            if desc_success and isinstance(desc_result, pd.DataFrame) and not desc_result.empty:
                columns = []
                for _, row in desc_result.iterrows():
                    columns.append({
                        "name": row["name"],
                        "type": row["type"],
                        "nullable": row.get("null?", "Y")
                    })
                view_info["columns"] = columns
        except Exception as e:
            logger.error(f"Error getting view columns: {str(e)}")
            view_info["columns"] = []
            
        return view_info
    
    def get_table_schema(self, table_name: str, database: Optional[str] = None, schema: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get schema information for a specific table
        
        Args:
            table_name: Table name
            database: Database name (uses default if None)
            schema: Schema name (uses default if None)
            
        Returns:
            List of column definitions with name, type, and nullable status
        """
        db = database or self.database
        sch = schema or self.schema
        if not db or not sch:
            return []
            
        query = f"DESCRIBE TABLE {db}.{sch}.{table_name}"
        success, result = self.execute_query(query)
        if success and isinstance(result, pd.DataFrame):
            columns = []
            for _, row in result.iterrows():
                columns.append({
                    "name": row["name"],
                    "type": row["type"],
                    "nullable": "Y" if row.get("null?", "Y") == "Y" else "N"
                })
            return columns
        return []
    
    def get_database_schema(self, database: Optional[str] = None, schema: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
        """
        Get schema information for all tables in a database
        
        Args:
            database: Database name (uses default if None)
            schema: Schema name (uses default if None)
            
        Returns:
            Dictionary mapping table names to their column definitions
        """
        tables = self.get_tables(database, schema)
        db_schema = {}
        
        for table in tables:
            db_schema[table] = self.get_table_schema(table, database, schema)
            
        return db_schema
    
    def get_table_sample(self, table_name: str, limit: int = 5, database: Optional[str] = None, schema: Optional[str] = None) -> pd.DataFrame:
        """
        Get a sample of data from a table
        
        Args:
            table_name: Table name
            limit: Maximum number of rows to return
            database: Database name (uses default if None)
            schema: Schema name (uses default if None)
            
        Returns:
            DataFrame with sample data
        """
        db = database or self.database
        sch = schema or self.schema
        if not db or not sch:
            return pd.DataFrame()
            
        query = f"SELECT * FROM {db}.{sch}.{table_name} LIMIT {limit}"
        success, result = self.execute_query(query)
        if success and isinstance(result, pd.DataFrame):
            return result
        return pd.DataFrame()
    
    def create_or_replace_view(
        self,
        view_name: str,
        sql_query: str,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        comment: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Create or replace a view in the database
        
        Args:
            view_name: Name for the view
            sql_query: SQL query for the view
            database: Database name (uses default if None)
            schema: Schema name (uses default if None)
            comment: Optional comment for the view
            
        Returns:
            Tuple of (success, message)
        """
        db = database or self.database
        sch = schema or self.schema
        if not db or not sch:
            return False, "No database or schema specified"
            
        # Construct the CREATE OR REPLACE VIEW statement
        create_view_sql = f"CREATE OR REPLACE VIEW {db}.{sch}.{view_name} "
        
        # Add comment if provided
        if comment:
            create_view_sql += f"COMMENT = '{comment}' "
            
        # Add query
        create_view_sql += f"AS {sql_query}"
        
        # Execute the CREATE VIEW statement
        success, result = self.execute_query(create_view_sql)
        
        if success:
            return True, f"View {db}.{sch}.{view_name} created successfully"
        else:
            return False, f"Error creating view: {result}"
    
    def format_schema_for_llm(self, database: Optional[str] = None, schema: Optional[str] = None) -> str:
        """
        Format database schema information for LLM context
        
        Args:
            database: Database name (uses default if None)
            schema: Schema name (uses default if None)
            
        Returns:
            Formatted schema information as string
        """
        db = database or self.database
        sch = schema or self.schema
        if not db or not sch:
            return "No database or schema information available."
            
        schema_info = self.get_database_schema(db, sch)
        if not schema_info:
            return f"No tables found in {db}.{sch}."
            
        formatted_schema = [f"Database: {db}", f"Schema: {sch}", "Tables:"]
        
        for table_name, columns in schema_info.items():
            formatted_schema.append(f"\nTABLE: {table_name}")
            formatted_schema.append("Columns:")
            
            for col in columns:
                nullable = "NULL" if col.get("nullable", "Y") == "Y" else "NOT NULL"
                formatted_schema.append(f"  - {col['name']} ({col['type']}) {nullable}")
        
        return "\n".join(formatted_schema)