import os
from typing import List, Dict, Any, Optional
from ingestion.offline_embeddings import OfflineEmbeddingProcessor

class OfflineSearch:
    """Offline multimodal search using FAISS index"""
    
    def __init__(self, db_path: str = "data/faiss_db"):
        """Initialize offline search"""
        try:
            self.processor = OfflineEmbeddingProcessor(db_path=db_path)
            print("OfflineSearch initialized successfully")
        except Exception as e:
            print(f"Failed to initialize OfflineSearch: {e}")
            self.processor = None
    
    def search(self, query: str, filters: Dict = None, top_k: int = 10, modality_filter: str = None) -> List[Dict[str, Any]]:
        """Perform multimodal search"""
        
        if not self.processor:
            print("Cannot perform search: FAISS processor not initialized")
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
            
            print(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            print(f"Search failed: {e}")
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