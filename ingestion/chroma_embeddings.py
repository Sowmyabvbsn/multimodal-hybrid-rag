import json
import os
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from PIL import Image
from loguru import logger
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


class ChromaEmbeddingProcessor:
    """Process chunks and store embeddings in ChromaDB"""
    
    def __init__(self, google_api_key: str, db_path: str = "data/chroma_db"):
        if not google_api_key:
            raise ValueError("Google API key is required")
            
        # Initialize Google AI
        genai.configure(api_key=google_api_key)
        self.google_client = genai
        
        # Initialize ChromaDB
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"ChromaDB initialized at: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
        
        # Initialize embedding models
        self._init_embedding_models()
        
        # Create collections
        if self.chroma_client:
            self._create_collections()
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", "."]
        )
    
    def _init_embedding_models(self):
        """Initialize embedding models"""
        try:
            # Use sentence-transformers for local embeddings
            self.dense_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {e}")
            self.dense_model = None
    
    def _create_collections(self):
        """Create ChromaDB collections"""
        if not self.chroma_client:
            logger.error("Cannot create collections: ChromaDB client not initialized")
            return
            
        try:
            # Create unified collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="unified_collection",
                metadata={"description": "Unified collection for text, tables, and images"}
            )
            logger.info("ChromaDB collection created successfully")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            self.collection = None
    
    def test_connection(self) -> bool:
        """Test if ChromaDB connection is working"""
        if not self.chroma_client:
            return False
            
        try:
            collections = self.chroma_client.list_collections()
            logger.info(f"ChromaDB connection test passed. Found {len(collections)} collections.")
            return True
        except Exception as e:
            logger.error(f"ChromaDB connection test failed: {e}")
            return False
    
    def create_image_summaries(self, chunks: List[Dict]) -> List[Dict]:
        """Create summaries for images using Gemini API"""
        image_chunks = []
        for chunk in chunks:
            if chunk["image"] and len(chunk['image']) > 0:
                for image_data in chunk["image"]:
                    try:
                        # Prepare prompt
                        section_header = image_data.get("section_header", "Section not available")
                        content = image_data.get("content", "Content not available")

                        prompt = f"""
                        Analyze this image and provide a comprehensive summary. 
                        Context: This image is from section "{section_header}" of a document with content {content}.
                        
                        Please summarize in **150 to 200 words max**:
                        1. A comprehensive description of what's shown in the image
                        2. Any text content visible in the image
                        3. Key visual elements, charts, diagrams, or data
                        4. How this relates to the section context
                        
                        Provide a clear, concise and detailed summary suitable for search and retrieval.
                        """
                        
                        image = Image.open(image_data["image_path"])
                        
                        # Generate summary using Gemini
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content([prompt, image])
                        summary = response.text
                        
                        metadata = {
                            "page_number": chunk["metadata"]["page_number"],
                            "document_id": chunk["metadata"]["document_id"],
                            "source_file": chunk["metadata"]["source_file"],
                            "image_path": image_data.get("image_path", ""),
                            "section_header": section_header,
                            "content": content,
                            "type": "image"
                        }
                        
                        # Create image chunk
                        image_chunk = {
                            "text": summary,
                            "metadata": metadata,
                        }

                        image_chunks.append(image_chunk)
                        
                        # Rate limiting
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Failed to process image: {e}")
                        continue
        
        logger.info(f"Created {len(image_chunks)} image summaries")
        return image_chunks
    
    def create_table_summaries(self, chunks: List[Dict]) -> List[Dict]:
        """Create summaries for tables using Gemini API"""
        table_chunks = []        
        for chunk in chunks:
            if chunk["table"] and len(chunk['table']) > 0:
                for table_data in chunk["table"]:
                    try:
                        # Prepare prompt
                        section_header = table_data.get("section_header", "")
                        content = table_data.get("content", "")
                        table_structure = table_data.get("table_data", "")
                        
                        prompt = f"""
                        Analyze this table and provide a comprehensive summary.
                        
                        Section Header: {section_header}
                        Table Content: {content}
                        Table Structure: {table_structure}
                        
                        Please summarize in **150 to 200 words max**:
                        1. What the table represents and its main purpose
                        2. Key data points, patterns, or insights
                        3. Notable values or totals
                        4. Why the table is significant in context

                        Keep the language clear and focused, suitable for search and retrieval.
                        """
                        
                        # Generate summary using Gemini
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(prompt)
                        summary = response.text

                        # Create metadata
                        metadata = {
                            "page_number": chunk["metadata"]["page_number"],
                            "document_id": chunk["metadata"]["document_id"],
                            "source_file": chunk["metadata"]["source_file"],
                            "section_header": section_header,
                            "content": table_structure,
                            "type": "table"
                        }

                        # Create table chunk
                        table_chunk = {
                            "text": summary,
                            "metadata": metadata,
                        }
                        
                        table_chunks.append(table_chunk)
                        
                        # Rate limiting
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Failed to process table: {e}")
                        continue
        
        logger.info(f"Created {len(table_chunks)} table summaries")
        return table_chunks
    
    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings for text using sentence-transformers"""
        if not self.dense_model:
            logger.error("Dense model not initialized")
            return np.array([])
            
        try:
            embeddings = self.dense_model.encode([text])
            return embeddings[0]
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return np.array([])
    
    def store_embeddings(self, chunks: List[Dict], chunk_type: str = "text"):
        """Process and store embeddings in ChromaDB"""
        if not chunks or not self.collection:
            return 0
        
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        num_of_chunks = 0

        for i, chunk in enumerate(chunks):
            if "text" not in chunk or not chunk["text"]:
                continue
                
            try:
                text = chunk["text"]
                # Clean metadata - remove None values and ensure all values are valid types
                metadata = self._clean_metadata(chunk["metadata"].copy())
                metadata["type"] = chunk_type
                
                if chunk_type == "text":
                    # Split long text into smaller chunks
                    text_chunks = self.splitter.split_text(text)
                    for j, sub_text in enumerate(text_chunks):
                        # Generate embeddings
                        embedding = self.generate_embeddings(sub_text)
                        
                        if len(embedding) == 0:
                            continue
                        
                        chunk_id = f"{chunk_type}_{i}_{j}_{uuid.uuid4().hex[:8]}"
                        
                        documents.append(sub_text)
                        metadatas.append(metadata)
                        ids.append(chunk_id)
                        embeddings.append(embedding.tolist())
                        num_of_chunks += 1
                else:
                    embedding = self.generate_embeddings(text)
                    if len(embedding) == 0:
                        continue
                    
                    chunk_id = f"{chunk_type}_{i}_{uuid.uuid4().hex[:8]}"
                    
                    documents.append(text)
                    metadatas.append(metadata)
                    ids.append(chunk_id)
                    embeddings.append(embedding.tolist())
                    num_of_chunks += 1
                
            except Exception as e:
                logger.error(f"Failed to process {chunk_type}: {e}")
                continue
        
        # Batch upload to ChromaDB
        if documents:
            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                logger.info(f"Stored {len(documents)} {chunk_type} embeddings")
            except Exception as e:
                logger.error(f"Failed to store embeddings: {e}")
                return 0

        return num_of_chunks

    def save_chunk_summaries(self, image_chunks: List[Dict], table_chunks: List[Dict], output_dir: str = "data/summaries"):
        """Save image and table summaries to JSON files for inspection."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if image_chunks:
            image_file = Path(output_dir) / "image_summaries.json"
            with open(image_file, "w", encoding="utf-8") as f:
                json.dump(image_chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(image_chunks)} image summaries → {image_file}")

        if table_chunks:
            table_file = Path(output_dir) / "table_summaries.json"
            with open(table_file, "w", encoding="utf-8") as f:
                json.dump(table_chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(table_chunks)} table summaries → {table_file}")

    def process_all_chunks(self, chunks: List[Dict]):
        """Process all chunks and store in ChromaDB collection"""
        if not self.chroma_client or not self.collection:
            logger.error("Cannot process chunks: ChromaDB not properly initialized")
            return {"text_chunks": 0, "image_chunks": 0, "table_chunks": 0, "error": "ChromaDB connection failed"}
            
        logger.info("Starting chunk processing...")
        
        # Process text
        num_of_text_chunks = self.store_embeddings(chunks, "text")

        # Process images
        image_chunks = self.create_image_summaries(chunks)
        num_of_image_chunks = self.store_embeddings(image_chunks, "image")

        # Process tables
        table_chunks = self.create_table_summaries(chunks)
        num_of_table_chunks = self.store_embeddings(table_chunks, "table")
        
        # Save summaries for inspection
        self.save_chunk_summaries(image_chunks, table_chunks, output_dir="data/summaries")

        logger.info("All chunks processed and stored in ChromaDB collection")
        
        result = {
            "text_chunks": num_of_text_chunks,
            "image_chunks": num_of_image_chunks,
            "table_chunks": num_of_table_chunks
        }

        return result

    def search(self, query: str, filters: Dict = None, top_k: int = 10) -> List[Dict]:
        """Search in ChromaDB collection"""
        if not self.collection:
            logger.error("Cannot search: Collection not initialized")
            return []
            
        try:
            # Build where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    # ChromaDB uses dot notation for nested metadata
                    where_clause[key] = value
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    score = 1 - distance
                    
                    result = {
                        "id": f"result_{i}",
                        "score": score,
                        "text": doc,
                        "type": metadata.get("type", "unknown"),
                        "source_file": metadata.get("source_file", "N/A"),
                        "page_number": metadata.get("page_number", "N/A"),
                        "section_header": metadata.get("section_header", ""),
                        "content": metadata.get("content", ""),
                        "image_path": metadata.get("image_path", "")
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _clean_metadata(self, metadata: Dict) -> Dict:
        """Clean metadata by removing None values and ensuring valid types"""
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                # Skip None values entirely
                continue
            elif isinstance(value, (str, int, float, bool)):
                # Keep valid primitive types
                cleaned[key] = value
            elif isinstance(value, (list, dict)):
                # Convert complex types to strings
                cleaned[key] = str(value)
            else:
                # Convert other types to string
                cleaned[key] = str(value)
        return cleaned

def main():
    """Example usage"""
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        return
    
    # Initialize processor
    processor = ChromaEmbeddingProcessor(
        google_api_key=google_api_key
    )
    
    # Test connection
    if processor.test_connection():
        logger.info("ChromaDB is ready to use!")
    else:
        logger.error("ChromaDB connection failed!")


if __name__ == "__main__":
    main()