from datetime import datetime
import json
import os
from pathlib import Path
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from PIL import Image
from loguru import logger
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter

# FastEmbed imports
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector
from ingestion.extract import PDFExtractor
from dotenv import load_dotenv
load_dotenv()


class QdrantEmbeddingProcessor:
    """Process chunks and store embeddings directly in Qdrant collections"""
    
    def __init__(self, google_api_key: str, qdrant_url: str, qdrant_api_key: str):
        # Initialize Google AI
        genai.configure(api_key=google_api_key)
        self.google_client = genai
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Initialize FastEmbed models
        self._init_fastembed_models()
        
        # Create collections
        # self._create_collections()
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", "."]
        )
    
    def _init_fastembed_models(self):
        """Initialize all FastEmbed models"""
        try:
            # self.dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
            self.dense_model = self.google_client
            self.sparse_model = SparseTextEmbedding("Qdrant/minicoil-v1")
            # self.rerank_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
        except Exception as e:
            logger.error(f"Failed to initialize FastEmbed models: {e}")
            raise
    
    def _create_collections(self):
        """Create Qdrant collections for text, tables, and images"""
        collections = ["text_collection", "tables_collection", "images_collection", "unified_collection"]
        
        for collection_name in collections:
            try:
                # Check if collection exists
                existing_collections = [col.name for col in self.qdrant_client.get_collections().collections]
                
                if collection_name not in existing_collections:
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "dense": VectorParams(size=3072, distance=Distance.COSINE)
                        },
                        sparse_vectors_config={
                            "sparse": SparseVectorParams()
                        }
                    )
                    logger.info(f"Created collection: {collection_name}")
                else:
                    logger.info(f"Collection {collection_name} already exists")
            except Exception as e:
                logger.error(f"Failed to create collection {collection_name}: {e}")
    
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
                            "content": table_structure
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
    
    def generate_embeddings(self, text: str) -> Dict[str, Any]:
        """Generate all types of embeddings for text"""
        embeddings = {}
        
        try:
            # Dense embeddings with google
            response = self.google_client.embed_content(
                model="gemini-embedding-001",
                content=text,
                output_dimensionality=3072,
                task_type="retrieval_document"
            )
            embeddings["dense"] = np.array(response['embedding'], dtype=np.float32)

            # Sparse embeddings (FastEmbed)
            sparse_emb = list(self.sparse_model.embed([text]))[0]
            embeddings["sparse"] = sparse_emb
            
            # # Reranking embeddings (FastEmbed)
            # rerank_emb = list(self.rerank_model.embed([text]))[0]
            # embeddings["rerank"] = rerank_emb
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return {}
    

    def store_embeddings(self, chunks: List[Dict], collection_name: str, chunk_type: str = "text"):
        """Process and store embeddings in Qdrant"""
        points = []
        base_uuid = uuid.uuid4()
        num_of_chunks = 0

        for i, chunk in enumerate(chunks):
            if "text" not in chunk or not chunk["text"]:
                continue
                
            try:
                text = chunk["text"]
                metadata = chunk["metadata"]
                metadata["type"] = chunk_type
                
                if chunk_type == "text":
                    # Split long text into smaller chunks
                    text_chunks = self.splitter.split_text(text)
                    for j, sub_text in enumerate(text_chunks):
                        # Generate embeddings
                        embeddings = self.generate_embeddings(sub_text)
                        
                        if not embeddings:
                            continue
                        
                        point_id = str(uuid.uuid5(base_uuid, str(j + 1)))
                        
                        # Create sparse vector
                        sparse_vector = SparseVector(
                            indices=list(embeddings["sparse"].indices),
                            values=list(embeddings["sparse"].values)
                        )

                        point = PointStruct(
                            id=point_id,
                            vector={"dense": embeddings["dense"].tolist(),
                                    "sparse": sparse_vector},
                            payload={"text": text, "metadata": metadata}
                        )
                        points.append(point)
                        num_of_chunks += 1
                else:
                    embeddings = self.generate_embeddings(text)
                    if not embeddings:
                        continue
                    
                    point_id = str(uuid.uuid5(base_uuid, str(i + 1)))
                    
                    # Create sparse vector
                    sparse_vector = SparseVector(
                        indices=list(embeddings["sparse"].indices),
                        values=list(embeddings["sparse"].values)
                    )

                    point = PointStruct(
                        id=point_id,
                        vector={"dense": embeddings["dense"].tolist(),
                                "sparse": sparse_vector},
                        payload={"text": text, "metadata": metadata}
                    )
                    points.append(point)
                    num_of_chunks += 1
                
            except Exception as e:
                logger.error(f"Failed to process {chunk_type}: {e}")
                continue
        
        # Batch upload to Qdrant
        if points:
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Stored {len(points)} {chunk_type} embeddings")

        return num_of_chunks

    def save_chunk_summaries(self, image_chunks: List[Dict], table_chunks: List[Dict], output_dir: str = "data/summaries"):
        """Save image and table summaries to JSON files for inspection."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if image_chunks:
            image_file = Path(output_dir) / "image_summaries.json"
            with open(image_file, "w", encoding="utf-8") as f:
                json.dump(image_chunks, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(image_chunks)} image summaries → {image_file}")

        if table_chunks:
            table_file = Path(output_dir) / "table_summaries.json"
            with open(table_file, "w", encoding="utf-8") as f:
                json.dump(table_chunks, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(table_chunks)} table summaries → {table_file}")

    def process_all_chunks(self, chunks: List[Dict]):
        """Process all chunks and store in appropriate collections"""
        logger.info("Starting chunk processing...")
        
        # Process text
        # self.store_embeddings(chunks, "text_collection", "text")
        num_of_text_chunks = self.store_embeddings(chunks, "unified_collection", "text")

        # Process images
        image_chunks = self.create_image_summaries(chunks)
        # self.store_embeddings(image_chunks, "images_collection", "image")
        num_of_image_chunks = self.store_embeddings(image_chunks, "unified_collection", "image")

        # Process tables
        table_chunks = self.create_table_summaries(chunks)
        # self.store_embeddings(table_chunks, "tables_collection", "table")
        num_of_table_chunks = self.store_embeddings(table_chunks, "unified_collection", "table")
        
        # Save summaries for inspection
        self.save_chunk_summaries(image_chunks, table_chunks, output_dir="data/summaries")

        logger.info("All chunks processed and stored in Qdrant collections")
        
        result = {
            "text_chunks": num_of_text_chunks,
            "image_chunks": num_of_image_chunks,
            "table_chunks": num_of_table_chunks
        }

        return result

def main():
    """Example usage"""
    
    # Initialize processor
    processor = QdrantEmbeddingProcessor(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY")
    )
    
    # Load your chunks data
    extractor = PDFExtractor()
    all_chunks = extractor.extract_all_pdfs()

    for pdf_path, chunks in all_chunks.items():
        processor.process_all_chunks(chunks)


if __name__ == "__main__":
    main()
