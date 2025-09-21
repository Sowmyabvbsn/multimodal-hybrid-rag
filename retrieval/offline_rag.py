import os
from typing import List, Dict, Any, Optional
from retrieval.offline_search import OfflineSearch
from retrieval.offline_llm import OfflineLLM

class OfflineRAG:
    """Complete offline multimodal RAG pipeline"""
    
    def __init__(self, db_path: str = "data/faiss_db"):
        """Initialize offline RAG system"""
        self.search = OfflineSearch(db_path=db_path)
        self.llm = OfflineLLM()
        
        print("OfflineRAG system initialized")
    
    def query(self, 
              question: str, 
              top_k: int = 5, 
              modality_filter: str = None,
              include_llm_response: bool = True) -> Dict[str, Any]:
        """Process a query through the complete RAG pipeline"""
        
        try:
            # Step 1: Retrieve relevant documents
            search_results = self.search.search(
                query=question,
                top_k=top_k,
                modality_filter=modality_filter
            )
            
            response = {
                'query': question,
                'search_results': search_results,
                'llm_response': None,
                'metadata': {
                    'total_results': len(search_results),
                    'modality_filter': modality_filter,
                    'top_k': top_k
                }
            }
            
            # Step 2: Generate LLM response if requested
            if include_llm_response and search_results:
                llm_response = self.llm.generate_response(question, search_results)
                response['llm_response'] = llm_response
            
            return response
            
        except Exception as e:
            print(f"RAG query failed: {e}")
            return {
                'query': question,
                'search_results': [],
                'llm_response': f"Sorry, I encountered an error: {str(e)}",
                'metadata': {'error': str(e)}
            }
    
    def cross_modal_query(self, question: str, top_k: int = 8) -> Dict[str, Any]:
        """Perform cross-modal query returning results from all modalities"""
        
        try:
            cross_results = self.search.cross_modal_search(question, top_k=top_k)
            
            # Flatten results for LLM processing
            all_results = []
            for modality, results in cross_results.items():
                all_results.extend(results)
            
            # Sort by score
            all_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Generate LLM response
            llm_response = None
            if all_results:
                llm_response = self.llm.generate_response(question, all_results[:top_k])
            
            return {
                'query': question,
                'cross_modal_results': cross_results,
                'combined_results': all_results[:top_k],
                'llm_response': llm_response,
                'metadata': {
                    'total_results': len(all_results),
                    'results_by_modality': {k: len(v) for k, v in cross_results.items()}
                }
            }
            
        except Exception as e:
            print(f"Cross-modal query failed: {e}")
            return {
                'query': question,
                'cross_modal_results': {},
                'combined_results': [],
                'llm_response': f"Sorry, I encountered an error: {str(e)}",
                'metadata': {'error': str(e)}
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        search_stats = self.search.get_statistics()
        
        return {
            'search_index': search_stats,
            'llm_model': self.llm.model_name if self.llm.model else "Template-based",
            'system_status': 'operational' if self.search.processor else 'degraded'
        }