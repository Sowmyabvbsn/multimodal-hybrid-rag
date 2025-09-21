import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import librosa
from loguru import logger
import time
from dotenv import load_dotenv

from ingestion.multimodal_extract import MultimodalChunk

load_dotenv()

class FAISSEmbeddingProcessor:
    """Process multimodal chunks and store embeddings in FAISS index"""
    
    def __init__(self, db_path: str = "data/faiss_db", embedding_dim: int = 384):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        
        # Initialize models
        self._init_models()
        
        # Initialize FAISS index
        self.index = None
        self.metadata_store = []
        self._init_faiss_index()
        
        # Load existing index if available
        self._load_existing_index()
    
    def _init_models(self):
        """Initialize embedding models"""
        try:
            # Text embeddings
            logger.info("Loading text embedding model...")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Image embeddings (CLIP)
            logger.info("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            logger.info("All embedding models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _init_faiss_index(self):
        """Initialize FAISS index"""
        try:
            # Use IndexFlatIP for cosine similarity (inner product after normalization)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info(f"Initialized FAISS index with dimension {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    def _load_existing_index(self):
        """Load existing FAISS index and metadata if available"""
        index_path = self.db_path / "faiss.index"
        metadata_path = self.db_path / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                with open(metadata_path, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load existing index: {e}")
                self._init_faiss_index()
                self.metadata_store = []
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            index_path = self.db_path / "faiss.index"
            metadata_path = self.db_path / "metadata.pkl"
            
            faiss.write_index(self.index, str(index_path))
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        try:
            embedding = self.text_model.encode([text], normalize_embeddings=True)
            return embedding[0]
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def generate_image_embedding(self, image_path: str, ocr_text: str = "") -> np.ndarray:
        """Generate embedding for image using CLIP and OCR text"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Generate CLIP embedding
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_embedding = image_features.numpy()[0]
            
            # If we have OCR text, combine with text embedding
            if ocr_text and ocr_text.strip():
                text_embedding = self.generate_text_embedding(ocr_text)
                # Weighted combination: 70% image, 30% text
                combined_embedding = 0.7 * image_embedding[:self.embedding_dim] + 0.3 * text_embedding
                # Normalize
                combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
                return combined_embedding
            else:
                # Resize and normalize image embedding to match dimension
                if len(image_embedding) != self.embedding_dim:
                    # Simple resize by truncation or padding
                    if len(image_embedding) > self.embedding_dim:
                        image_embedding = image_embedding[:self.embedding_dim]
                    else:
                        padding = np.zeros(self.embedding_dim - len(image_embedding))
                        image_embedding = np.concatenate([image_embedding, padding])
                
                return image_embedding / np.linalg.norm(image_embedding)
                
        except Exception as e:
            logger.error(f"Failed to generate image embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def generate_audio_embedding(self, transcript: str) -> np.ndarray:
        """Generate embedding for audio using transcript"""
        # For now, use text embedding of transcript
        # Could be enhanced with audio feature extraction
        return self.generate_text_embedding(transcript)
    
    def process_chunk(self, chunk: MultimodalChunk) -> Optional[np.ndarray]:
        """Process a single chunk and generate embedding"""
        try:
            if chunk.chunk_type == 'text':
                return self.generate_text_embedding(chunk.content)
            
            elif chunk.chunk_type == 'image':
                if chunk.image_path and os.path.exists(chunk.image_path):
                    return self.generate_image_embedding(chunk.image_path, chunk.ocr_text or "")
                else:
                    # Fallback to text embedding if image not available
                    return self.generate_text_embedding(chunk.content)
            
            elif chunk.chunk_type == 'audio':
                if chunk.transcript:
                    return self.generate_audio_embedding(chunk.transcript)
                else:
                    return self.generate_text_embedding(chunk.content)
            
            elif chunk.chunk_type == 'table':
                return self.generate_text_embedding(chunk.content)
            
            else:
                logger.warning(f"Unknown chunk type: {chunk.chunk_type}")
                return self.generate_text_embedding(chunk.content)
                
        except Exception as e:
            logger.error(f"Failed to process chunk: {e}")
            return None
    
    def add_chunks(self, chunks: List[MultimodalChunk]) -> int:
        """Add chunks to FAISS index"""
        embeddings = []
        valid_chunks = []
        
        logger.info(f"Processing {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            embedding = self.process_chunk(chunk)
            if embedding is not None and not np.isnan(embedding).any():
                embeddings.append(embedding)
                valid_chunks.append(chunk)
            else:
                logger.warning(f"Skipping chunk due to invalid embedding: {chunk.source_file}")
        
        if not embeddings:
            logger.warning("No valid embeddings generated")
            return 0
        
        # Convert to numpy array and add to index
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings_array)
        
        # Store metadata
        for chunk in valid_chunks:
            metadata = {
                'content': chunk.content,
                'chunk_type': chunk.chunk_type,
                'source_file': chunk.source_file,
                'page_number': chunk.page_number,
                'timestamp': chunk.timestamp,
                'section_header': chunk.section_header,
                'image_path': chunk.image_path,
                'audio_path': chunk.audio_path,
                'ocr_text': chunk.ocr_text,
                'transcript': chunk.transcript,
                'metadata': chunk.metadata
            }
            self.metadata_store.append(metadata)
        
        # Save index
        self._save_index()
        
        logger.info(f"Added {len(valid_chunks)} chunks to FAISS index")
        return len(valid_chunks)
    
    def search(self, query: str, top_k: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search in FAISS index"""
        try:
            if self.index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return []
            
            # Generate query embedding
            query_embedding = self.generate_text_embedding(query)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search in FAISS
            scores, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                
                metadata = self.metadata_store[idx]
                
                # Apply filters if provided
                if filters:
                    skip = False
                    for key, value in filters.items():
                        if key in metadata and metadata[key] != value:
                            skip = True
                            break
                    if skip:
                        continue
                
                result = {
                    'id': idx,
                    'score': float(score),
                    'text': metadata['content'],
                    'type': metadata['chunk_type'],
                    'source_file': metadata['source_file'],
                    'page_number': metadata['page_number'],
                    'timestamp': metadata['timestamp'],
                    'section_header': metadata['section_header'],
                    'image_path': metadata['image_path'],
                    'audio_path': metadata['audio_path'],
                    'ocr_text': metadata['ocr_text'],
                    'transcript': metadata['transcript'],
                    'metadata': metadata['metadata']
                }
                results.append(result)
                
                if len(results) >= top_k:
                    break
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = {
            'total_vectors': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.embedding_dim,
            'index_size_mb': 0
        }
        
        # Calculate index size
        index_path = self.db_path / "faiss.index"
        if index_path.exists():
            stats['index_size_mb'] = index_path.stat().st_size / (1024 * 1024)
        
        # Count by type
        type_counts = {}
        for metadata in self.metadata_store:
            chunk_type = metadata['chunk_type']
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        stats['type_distribution'] = type_counts
        
        return stats

def main():
    """Example usage"""
    from ingestion.multimodal_extract import MultimodalExtractor
    
    # Extract content
    extractor = MultimodalExtractor()
    all_chunks = extractor.extract_all_files()
    
    # Initialize FAISS processor
    processor = FAISSEmbeddingProcessor()
    
    # Process all chunks
    total_added = 0
    for file_path, chunks in all_chunks.items():
        added = processor.add_chunks(chunks)
        total_added += added
        logger.info(f"Added {added} chunks from {Path(file_path).name}")
    
    # Print statistics
    stats = processor.get_stats()
    logger.info(f"FAISS Index Statistics: {stats}")
    
    # Test search
    if total_added > 0:
        results = processor.search("sentence embeddings", top_k=5)
        logger.info(f"Test search returned {len(results)} results")

if __name__ == "__main__":
    main()