import os
from typing import List, Dict, Any, Optional
from loguru import logger
from retrieval.multimodal_search import MultimodalSearch
from retrieval.offline_llm import OfflineLLM
from dotenv import load_dotenv

load_dotenv()

class MultimodalRAG:
    """Complete multimodal RAG pipeline"""
    
    def __init__(self, db_path: str = "data/faiss_db"):
        """Initialize multimodal RAG system"""
        self.search = MultimodalSearch(db_path=db_path)
        self.llm = OfflineLLM()
        
        logger.info("MultimodalRAG system initialized")
    
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
            logger.error(f"RAG query failed: {e}")
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
            logger.error(f"Cross-modal query failed: {e}")
            return {
                'query': question,
                'cross_modal_results': {},
                'combined_results': [],
                'llm_response': f"Sorry, I encountered an error: {str(e)}",
                'metadata': {'error': str(e)}
            }
    
    def batch_query(self, questions: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Process multiple queries in batch"""
        results = []
        
        for question in questions:
            result = self.query(question, top_k=top_k)
            results.append(result)
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        search_stats = self.search.get_statistics()
        
        return {
            'search_index': search_stats,
            'llm_model': self.llm.model_name if self.llm.model else "Template-based",
            'system_status': 'operational' if self.search.processor else 'degraded'
        }
    
    def export_results(self, results: Dict[str, Any], format: str = 'json') -> str:
        """Export query results to different formats"""
        
        if format.lower() == 'json':
            import json
            return json.dumps(results, indent=2, ensure_ascii=False)
        
        elif format.lower() == 'text':
            output = []
            output.append(f"Query: {results['query']}")
            output.append("=" * 50)
            
            if results.get('llm_response'):
                output.append("AI Response:")
                output.append(results['llm_response'])
                output.append("")
            
            output.append("Search Results:")
            for i, result in enumerate(results['search_results'], 1):
                output.append(f"\n{i}. {result['type'].upper()} - {result['source_file']}")
                if result.get('page_number'):
                    output.append(f"   Page: {result['page_number']}")
                if result.get('timestamp'):
                    output.append(f"   Time: {result['timestamp']:.1f}s")
                output.append(f"   Score: {result['score']:.4f}")
                output.append(f"   Content: {result['text'][:200]}...")
            
            return "\n".join(output)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")

def main():
    """Test multimodal RAG system"""
    rag = MultimodalRAG()
    
    # Test single query
    print("Testing single query...")
    result = rag.query("What is machine learning?", top_k=3)
    
    print(f"Query: {result['query']}")
    print(f"Found {len(result['search_results'])} results")
    
    if result['llm_response']:
        print(f"\nAI Response:\n{result['llm_response']}")
    
    print("\nSearch Results:")
    for i, res in enumerate(result['search_results'], 1):
        print(f"{i}. {res['type']} - {res['source_file']} (Score: {res['score']:.4f})")
    
    # Test cross-modal query
    print("\n" + "="*50)
    print("Testing cross-modal query...")
    cross_result = rag.cross_modal_query("neural networks", top_k=6)
    
    print(f"Cross-modal results:")
    for modality, results in cross_result['cross_modal_results'].items():
        print(f"  {modality}: {len(results)} results")
    
    # Print system stats
    print("\n" + "="*50)
    print("System Statistics:")
    stats = rag.get_system_stats()
    print(f"Search Index: {stats['search_index']}")
    print(f"LLM Model: {stats['llm_model']}")
    print(f"Status: {stats['system_status']}")

if __name__ == "__main__":
    main()