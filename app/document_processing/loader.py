"""
Document loading utilities for different file types
"""
import os
from typing import List, Dict, Any, Union
import pandas as pd
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader


class DocumentLoader:
    """Handles loading documents of various formats"""
    
    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        """
        Load a document based on its file extension
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        
        Raises:
            ValueError: If file type is not supported
        """
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        if file_extension == '.txt':
            return DocumentLoader._load_text(file_path)
        elif file_extension == '.csv':
            return DocumentLoader._load_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return DocumentLoader._load_excel(file_path)
        elif file_extension == '.json':
            return DocumentLoader._load_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    @staticmethod
    def _load_text(file_path: str) -> List[Document]:
        """Load a text file"""
        loader = TextLoader(file_path)
        return loader.load()
    
    @staticmethod
    def _load_csv(file_path: str) -> List[Document]:
        """Load a CSV file"""
        loader = CSVLoader(file_path)
        return loader.load()
    
    @staticmethod
    def _load_excel(file_path: str) -> List[Document]:
        """
        Load an Excel file
        Converts each sheet to a Document
        """
        documents = []
        excel_file = pd.ExcelFile(file_path)
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            # Convert dataframe to text
            text = df.to_string(index=False)
            metadata = {
                "source": file_path,
                "sheet_name": sheet_name
            }
            documents.append(Document(page_content=text, metadata=metadata))
        
        return documents
    
    @staticmethod
    def _load_json(file_path: str) -> List[Document]:
        """
        Load a JSON file
        """
        import json
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # If it's a list of objects, create a document for each
        if isinstance(data, list):
            documents = []
            for i, item in enumerate(data):
                text = json.dumps(item, indent=2)
                metadata = {
                    "source": file_path,
                    "item_index": i
                }
                documents.append(Document(page_content=text, metadata=metadata))
            return documents
            
        # If it's a single object, create one document
        text = json.dumps(data, indent=2)
        metadata = {"source": file_path}
        return [Document(page_content=text, metadata=metadata)]
    
    @staticmethod
    def convert_to_string(data: Union[pd.DataFrame, Dict, List, Any]) -> str:
        """
        Convert various data types to a string representation
        
        Args:
            data: Data to convert (DataFrame, dict, list, etc.)
            
        Returns:
            String representation of the data
        """
        if isinstance(data, pd.DataFrame):
            return data.to_string(index=False)
        elif isinstance(data, (dict, list)):
            import json
            return json.dumps(data, indent=2)
        else:
            return str(data)
