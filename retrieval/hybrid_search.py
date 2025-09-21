# pdf_rag_pipeline/retrieval/hybrid_search.py

import os
from typing import List, Dict, Any
from loguru import logger
import numpy as np
from ingestion.vector_embeddings import QdrantEmbeddingProcessor
from qdrant_client.models import SparseVector, Prefetch, Filter, FieldCondition, MatchValue

from dotenv import load_dotenv
load_dotenv()

class HybridSearch:
    def __init__(self, collection_name: str = "unified_collection"):
        """Initialize hybrid search with weights."""
        self.collection_name = collection_name
        
        # Initialize processor
        try:
            self.processor = QdrantEmbeddingProcessor(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                qdrant_url=os.getenv("QDRANT_URL"),
                qdrant_api_key=os.getenv("QDRANT_API_KEY")
            )
            
            # Only proceed if Qdrant connection is successful
            if self.processor.qdrant_client:
                # Create collection and indexes
                self._ensure_collection_exists()
                self._create_payload_indexes()
                
                self.dense_model = self.processor.dense_model
                self.sparse_model = self.processor.sparse_model
                
                logger.info("HybridSearch initialized successfully")
            else:
                logger.error("HybridSearch initialization failed: No Qdrant connection")
                
        except Exception as e:
            logger.error(f"Failed to initialize HybridSearch: {e}")
            self.processor = None
        
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if it doesn't"""
        if not self.processor or not self.processor.qdrant_client:
            logger.error("Cannot ensure collection exists: No Qdrant connection")
            return
        
        try:
            existing_collections = [col.name for col in self.processor.qdrant_client.get_collections().collections]
            if self.collection_name not in existing_collections:
                self.processor._create_collections()
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
    
    def _create_payload_indexes(self):
        """Create payload indexes for efficient filtering"""
        if not self.processor or not self.processor.qdrant_client:
            logger.error("Cannot create payload indexes: No Qdrant connection")
            return
        
        try:
            from qdrant_client.http import models as rest
            
            # Check if collection exists before creating indexes
            existing_collections = [col.name for col in self.processor.qdrant_client.get_collections().collections]
            if self.collection_name not in existing_collections:
                logger.warning(f"Collection {self.collection_name} doesn't exist, skipping index creation")
                return
            
            # Create indexes for efficient filtering
            indexes_to_create = [
                ("metadata.type", rest.PayloadSchemaType.KEYWORD),
                ("metadata.source_file", rest.PayloadSchemaType.KEYWORD),
                ("metadata.page_number", rest.PayloadSchemaType.INTEGER)
            ]
            
            for field_name, schema_type in indexes_to_create:
                try:
                    self.processor.qdrant_client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema_type
                    )
                    logger.info(f"Created index for {field_name}")
                except Exception as e:
                    # Index might already exist, which is fine
                    if "already exists" in str(e).lower():
                        logger.info(f"Index for {field_name} already exists")
                    else:
                        logger.warning(f"Failed to create index for {field_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error creating payload indexes: {e}")

    def build_filter(self, filters: Dict) -> Filter | None:
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            conditions.append(FieldCondition(
                key=key,
                match=MatchValue(value=value)
            ))
        return Filter(must=conditions)
    
    def search(self, query: str, filters: Dict, top_k: int = 10, result_type: str = "dense", collection_name: str = None) -> List[Dict[str, Any]]:
        """Perform hybrid search and return ranked results."""
        
        if not self.processor or not self.processor.qdrant_client:
            logger.error("Cannot perform search: No Qdrant connection")
            return []
        
        if collection_name is None:
            collection_name = self.collection_name
        
        # Generate Embeddings
        try:
            query_embeddings = self.processor.generate_embeddings(query)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []

        sparse_vector = SparseVector(
                        indices=list(query_embeddings["sparse"].indices),
                        values=list(query_embeddings["sparse"].values)
                    )
        dense_embeddings = query_embeddings['dense'].tolist()

        query_filters = self.build_filter(filters)

        try:
            if result_type == "dense":
                # 1. Dense search
                dense_results = self.processor.qdrant_client.query_points(
                    collection_name=collection_name,
                    query=dense_embeddings,
                    using="dense",
                    limit=top_k,
                    query_filter=query_filters,
                    with_payload=True
                )

                return dense_results

            elif result_type == "sparse":
                # 2. Sparse search
                sparse_results = self.processor.qdrant_client.query_points(
                    collection_name=collection_name,
                    query=sparse_vector,
                    using="sparse",
                    limit=top_k,
                    query_filter=query_filters,
                    with_payload=True
                )

                return sparse_results

            # 3. Standard hybrid searches

            elif result_type == "hybrid_dense":
            # Retrieve candidates from Qdrant
                hybrid_dense = self.processor.qdrant_client.query_points(
                    collection_name=collection_name,
                    query=dense_embeddings,
                    using="dense",
                    prefetch=[
                        Prefetch(query=sparse_vector, using="sparse"),
                    ],
                    limit=top_k,
                    with_payload=True,
                    query_filter=query_filters
                )
                return hybrid_dense
            
            elif result_type == "hybrid_sparse":
                hybrid_sparse = self.processor.qdrant_client.query_points(
                    collection_name=collection_name,
                    query=sparse_vector,
                    using="sparse",
                    prefetch=[
                        Prefetch(query=dense_embeddings, using="dense"),
                    ],
                    limit=top_k,
                    with_payload=True,
                    query_filter=query_filters
                )
                return hybrid_sparse

            else:
                raise ValueError(f"Invalid result_type '{result_type}'. Choose from: dense, sparse, hybrid_dense, hybrid_sparse.")
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
        
    def process_results(self, results):
        """
        Process hybrid search results into a list of dictionaries.
        Each dictionary contains id, score, text, type, source_file, 
        page_number, section_header, content, and image_path.
        """
        processed = []

        for points, score_card in results:
            for result in score_card:
                payload = result.payload
                metadata = payload.get("metadata", {})

                entry = {
                    "id": result.id,
                    "score": result.score,
                    "text": payload.get("text"),
                    "type": metadata.get("type"),
                    "source_file": metadata.get("source_file"),
                    "page_number": metadata.get("page_number"),
                    "section_header": metadata.get("section_header",""),
                    "content": metadata.get("content",""),
                    "image_path": metadata.get("image_path","")
                }
                processed.append(entry)
                
        return processed
    
def main():
    # pass
    hybrid_search = HybridSearch()
    # results = hybrid_search.search("Nordic Semiconducto", top_k=5, filters={"metadata.type": "text", "metadata.source_file": "test_pdf2.pdf"}, result_type="hybrid_dense")
    # print(results)
    # processed_results = hybrid_search.process_results(results)
    # print(processed_results)


if __name__ == "__main__":
    main()


    # # 3. Combine with alpha weighting
    # scores = {}

    # for res in dense_results:
    #     scores[res.id] = alpha * res.score

    # for res in sparse_results:
    #     scores[res.id] = scores.get(res.id, 0) + (1 - alpha) * res.score

    # # 4. Rerank
    # reranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # return reranked[:top_k]