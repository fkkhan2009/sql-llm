import asyncio 
import asyncpg 
from typing import Optional, Dict, List, Any
import os 
import logging
from fastapi import FastAPI, APIRouter
from contextlib import asynccontextmanager
from pydantic import BaseModel
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

venv_site_packages_path = '/Users/kakhan/Documents/Coding_workspace/python/llm/llm-poc/.venv/lib/python3.12/site-packages'
sys.path.append(venv_site_packages_path)

logger = logging.getLogger(__name__)


# from app.config import settings

load_dotenv(".env.learn")


# class PostgresConnector:
#     def __init__(self,
#                  host: str =  os.getenv("POSTGRES_HOST"),
#                  user: str = os.getenv("POSTGRES_USER"),
#                  password: str = os.getenv("POSTGRES_PASSWORD"),
#                  database: str =  os.getenv("POSTGRES_DB"),
#                  port: int =  os.getenv("POSTGRES_PORT", 5432)):
       
#         # initialie and set the connnecton param
#        self.host = host 
#        self.user = user 
#        self.password = password 
#        self.database = database 
#        self.port = port

#        if not self.host or not self.user or not self.password or not self.database:
#         raise ValueError("Missing required connection parameters for Psotgres")
#        self.conn = None
       

#     async def connect(self):
#        # make a connection to the postgres db 
#        try:
#           self.conn = await asyncpg.connect(
#              host=self.host,
#              user=self.user,
#              password=self.password,
#              database=self.database,
#              port=self.port
#           )
#        except ConnectionError as e:
#           raise ConnectionError(f"Failed to connect to postgres: {e}")
       
#     async def execute_query(self, query: str):
#        try:
#           if self.conn:
#              # execute the query 
#              results = await self.conn.execute(query)
#        except Exception as e:
#           logger.error(f"Error executing query: {e}")

#     async def close_connection(self):
#        try:
#           await self.conn.close()
#        except Exception as e:
#           logger.error(f"Error closing connection: {e}")
        



    

router = APIRouter(
       prefix="/postgres",
       tags=["postgres"])

class TestRequest(BaseModel):
   query: Optional[str] = None

class TestResponse(BaseModel):
   results: Optional[List[Dict[str, Any]]] = None

   


# pgconnector = None

# @asynccontextmanager
# async def lifespan(app: FastAPI):

#    global pgconnector
#    load_dotenv(".env.learn")
#    pgconnector = PostgresConnector()
#    logger.info('Connecting to Postgres...')
#    await pgconnector.connect()
#    logger.info('Connected to Postgres')
#    yield 
#    logger.info('Closing connection to Postgres...')
#    await pgconnector.close_connection()
#    logger.info('Connection to Postgres closed')



   
# @router.post("/test")
# async def test_connection(request: TestRequest):
#    global pgconnector
#    results = await pgconnector.execute_query(request.query)

#    return 
   

# @router.get("/multiple_test")
# async def test_connection():
#    global pgconnector
#    results = await pgconnector.execute_query("INSERT INTO test VALUES (1, 'test')")

#    return 

# app= FastAPI(
#    lifespan=lifespan
# )

# app.include_router(router)
          
# if __name__ == "__main__":
   
#    import uvicorn
#    uvicorn.run("postgres_connector:app", host="0.0.0.0", port=8081, reload=True)
       


    

    
       
       
       
        

class PostgresConnectorSync:
    def __init__(self,
                 host: str = os.getenv("POSTGRES_HOST"),
                 user: str = os.getenv("POSTGRES_USER"),
                 password: str = os.getenv("POSTGRES_PASSWORD"),
                 database: str = os.getenv("POSTGRES_DB"),
                 port: int = os.getenv("POSTGRES_PORT", 5432)):
        # Initialize and set the connection parameters
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port

        if not self.host or not self.user or not self.password or not self.database:
            raise ValueError("Missing required connection parameters for Postgres")
        self.conn = None

    def connect(self):
        # Make a connection to the postgres db
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )
            logger.info('Connected to Postgres')
        except Exception as e:
            logger.error(f"Failed to connect to postgres: {e}")
            raise

    def execute_query(self, query: str):
        try:
            if self.conn:
                with self.conn.cursor() as cursor:
                    cursor.execute(query)
                    self.conn.commit()  # Commit the transaction
                    logger.info(f"Executed query: {query}")
        except Exception as e:
            logger.error(f"Error executing query: {e}")

    def close_connection(self):
        try:
            if self.conn:
                self.conn.close()
                logger.info('Connection to Postgres closed')
        except Exception as e:
            logger.error(f"Error closing connection: {e}")

pgconnector_sync = PostgresConnectorSync()


@router.get("/multiple_test_sync")
def test_connection():
    global pgconnector_sync
    pgconnector_sync.execute_query(("INSERT INTO test VALUES (1, 'test')"))
    return {"message": "Query executed successfully"}

app = FastAPI()

app.include_router(router)

@app.on_event("startup")
def lifespan():
    global pgconnector_sync
    pgconnector_sync.connect()
    
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("postgres_connector:app", host="0.0.0.0", port=8081, reload=True)
       


    

    
       
       
       
        