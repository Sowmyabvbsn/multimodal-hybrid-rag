import os
from typing import List, Dict, Any, Optional
from loguru import logger
from ingestion.faiss_embeddings import FAISSEmbeddingProcessor
from dotenv import load_dotenv

load_dotenv()

class MultimodalSearch:
    """Multimodal search using FAISS index"""
    
    def __init__(self, db_path: str = "data/faiss_db"):
        """Initialize multimodal search"""
        try:
            self.processor = FAISSEmbeddingProcessor(db_path=db_path)
            logger.info("MultimodalSearch initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MultimodalSearch: {e}")
            self.processor = None
    
    def search(self, query: str, filters: Dict = None, top_k: int = 10, modality_filter: str = None) -> List[Dict[str, Any]]:
        """Perform multimodal search"""
        
        if not self.processor:
            logger.error("Cannot perform search: FAISS processor not initialized")
            return []
        
        try:
            # Apply modality filter
            search_filters = filters.copy() if filters else {}
            if modality_filter and modality_filter.lower() != 'all':
                search_filters['chunk_type'] = modality_filter.lower()
            
            results = self.processor.search(
                query=query,
                top_k=top_k,
                filters=search_filters
            )
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search index statistics"""
        if not self.processor:
            return {}
        
        return self.processor.get_stats()
    
    def search_by_modality(self, query: str, modality: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search within specific modality"""
        return self.search(query, modality_filter=modality, top_k=top_k)
    
    def cross_modal_search(self, query: str, top_k: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Perform cross-modal search returning results grouped by modality"""
        results = {
            'text': [],
            'image': [],
            'audio': [],
            'table': []
        }
        
        # Search each modality
        for modality in results.keys():
            modality_results = self.search_by_modality(query, modality, top_k=top_k//4 + 1)
            results[modality] = modality_results
        
        return results

def main():
    """Test multimodal search"""
    search = MultimodalSearch()
    
    if search.processor:
        # Test search
        results = search.search("machine learning", top_k=5)
        
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Score: {result['score']:.4f}")
            print(f"Type: {result['type']}")
            print(f"Source: {result['source_file']}")
            print(f"Content: {result['text'][:200]}...")
        
        # Test cross-modal search
        cross_results = search.cross_modal_search("neural networks", top_k=8)
        print(f"\nCross-modal search results:")
        for modality, results in cross_results.items():
            print(f"{modality}: {len(results)} results")
        
        # Print statistics
        stats = search.get_statistics()
        print(f"\nIndex Statistics: {stats}")
    else:
        print("MultimodalSearch initialization failed")

if __name__ == "__main__":
    main()