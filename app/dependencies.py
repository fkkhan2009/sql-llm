"""
Updated dependency injection with LLM provider support
"""
from typing import Dict, Any
from functools import lru_cache

from app.document_processing.processor import DocumentProcessor
from app.document_processing.vector_store import VectorStore
from app.database.snowflake_connector import SnowflakeConnector
from app.database.schema_utils import SchemaUtils
from app.llm.client import LLMClient
from app.llm.provider_factory import LLMProviderFactory
from app.llm.context_builder import ContextBuilder
from app.llm.orchestrator import LLMOrchestrator
from app.llm.enhanced_orchestrator import EnhancedLLMOrchestrator
from app.workspace.manager import WorkspaceManager
from app.conversation.manager import ConversationManager

from app.config import settings


class AppComponents:
    """Container for application components"""
    
    def __init__(self):
        """Initialize application components"""
        self.initialized = False
        self.components = {}
    
    def initialize(self):
        """Initialize all components"""
        if self.initialized:
            return
        
        # Initialize basic components
        self.vector_store = VectorStore()
        self.document_processor = DocumentProcessor()
        
        # Initialize LLM client with selected provider
        try:
            llm_provider = LLMProviderFactory.create_provider(
                provider_type=settings.LLM_PROVIDER,
                config=settings.get_llm_config()
            )
            self.llm_client = LLMClient(provider=llm_provider)
            
            # Log provider information
            provider_info = self.llm_client.provider_info
            print(f"Initialized LLM provider: {provider_info['provider_name']}")
            print(f"Model: {provider_info['model_name']}")
            print(f"Embedding model: {provider_info['embedding_model']}")
            print(f"Context window: {provider_info['context_window']} tokens")
        except Exception as e:
            print(f"Failed to initialize LLM client: {str(e)}")
            self.llm_client = None
        
        # Initialize workspace and conversation management
        self.workspace_manager = WorkspaceManager()
        if self.llm_client:
            self.conversation_manager = ConversationManager(
                workspace_manager=self.workspace_manager,
                llm_client=self.llm_client
            )
        else:
            self.conversation_manager = None
        
        # Initialize Snowflake connector if credentials are available
        self.snowflake_connector = None
        self.schema_utils = None
        if settings.CONNECT_TO_DB:
            if (settings.SNOWFLAKE_ACCOUNT and settings.SNOWFLAKE_USER and 
                (settings.SNOWFLAKE_PASSWORD or settings.SNOWFLAKE_PASSWORD_FILE or 
                settings.SNOWFLAKE_PRIVATE_KEY_FILE)):
                try:
                    self.snowflake_connector = SnowflakeConnector()
                    # Test connection
                    self.snowflake_connector.connect()
                    print("Successfully connected to Snowflake")
                    
                    # Initialize schema utils
                    self.schema_utils = SchemaUtils(self.snowflake_connector)
                except Exception as e:
                    print(f"Failed to initialize Snowflake connector: {str(e)}")
                    self.snowflake_connector = None
                    self.schema_utils = None
            else:
                self.schema_utils = None
        
        # Initialize context builder and orchestrator
        if self.llm_client:
            self.context_builder = ContextBuilder(
                vector_store=self.vector_store,
                snowflake_connector=self.snowflake_connector
            )
            
            self.llm_orchestrator = LLMOrchestrator(
                llm_client=self.llm_client,
                context_builder=self.context_builder,
                vector_store=self.vector_store,
                snowflake_connector=self.snowflake_connector
            )
            
            # Initialize enhanced orchestrator
            if self.snowflake_connector and self.schema_utils and settings.CONNECT_TO_DB:
                self.enhanced_orchestrator = EnhancedLLMOrchestrator(
                    llm_client=self.llm_client,
                    snowflake_connector=self.snowflake_connector,
                    schema_utils=self.schema_utils,
                    context_builder=self.context_builder,
                    vector_store=self.vector_store
                )
            else:
                self.enhanced_orchestrator = EnhancedLLMOrchestrator(
                    llm_client=self.llm_client,
                    #snowflake_connector=self.snowflake_connector,
                    #schema_utils=self.schema_utils,
                    context_builder=self.context_builder,
                    vector_store=self.vector_store
                )
        else:
            self.context_builder = None
            self.llm_orchestrator = None
            self.enhanced_orchestrator = None
        
        # Mark as initialized
        self.initialized = True
    
    def get_components(self) -> Dict[str, Any]:
        """Get all components as a dictionary"""
        if not self.initialized:
            self.initialize()
            
        return {
            "document_processor": self.document_processor,
            "vector_store": self.vector_store,
            "snowflake_connector": self.snowflake_connector,
            "schema_utils": self.schema_utils,
            "llm_client": self.llm_client,
            "context_builder": self.context_builder,
            "llm_orchestrator": self.llm_orchestrator,
            "enhanced_orchestrator": self.enhanced_orchestrator,
            "workspace_manager": self.workspace_manager,
            "conversation_manager": self.conversation_manager
        }


# Create a singleton instance
app_components = AppComponents()


# FastAPI dependency - this is what routers will use
@lru_cache()
def get_components():
    """Get all application components"""
    app_components.initialize()
    return app_components.get_components()


# Enhanced components for new router
def get_enhanced_components():
    """Get components for enhanced database queries"""
    components = get_components()
    
    # Check if enhanced orchestrator is available
    #if not components["enhanced_orchestrator"]:
        #raise ValueError("Enhanced orchestrator is not available. Please check Snowflake connection.")
        
    return components


# Individual component dependencies
def get_document_processor():
    """Get document processor component"""
    components = get_components()
    return components["document_processor"]


def get_vector_store():
    """Get vector store component"""
    components = get_components()
    return components["vector_store"]


def get_snowflake_connector():
    """Get Snowflake connector component"""
    components = get_components()
    return components["snowflake_connector"]


def get_schema_utils():
    """Get schema utils component"""
    components = get_components()
    return components["schema_utils"]


def get_llm_client():
    """Get LLM client component"""
    components = get_components()
    return components["llm_client"]


def get_llm_orchestrator():
    """Get LLM orchestrator component"""
    components = get_components()
    return components["llm_orchestrator"]


def get_enhanced_orchestrator():
    """Get enhanced LLM orchestrator component"""
    components = get_components()
    return components["enhanced_orchestrator"]


def get_workspace_manager():
    """Get workspace manager component"""
    components = get_components()
    return components["workspace_manager"]


def get_conversation_manager():
    """Get conversation manager component"""
    components = get_components()
    return components["conversation_manager"]


# Specific component combinations for routers
def get_base_components():
    """Get components for base router"""
    components = get_components()
    return components


def get_workspace_components():
    """Get components for workspace router"""
    components = get_components()
    return {
        "document_processor": components["document_processor"],
        "workspace_manager": components["workspace_manager"]
    }


def get_conversation_components():
    """Get components for conversation router"""
    components = get_components()
    return {
        "workspace_manager": components["workspace_manager"],
        "conversation_manager": components["conversation_manager"],
        "snowflake_connector": components["snowflake_connector"],
        "llm_client": components["llm_client"]
    }