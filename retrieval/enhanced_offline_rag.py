import os
from typing import List, Dict, Any, Optional
from retrieval.offline_search import OfflineSearch
from retrieval.enhanced_llm import EnhancedOfflineLLM

class EnhancedOfflineRAG:
    """Enhanced offline multimodal RAG pipeline with robust LLM integration"""
    
    def __init__(self, db_path: str = "data/faiss_db", llm_model: str = "microsoft/DialoGPT-medium"):
        """Initialize enhanced offline RAG system"""
        self.search = OfflineSearch(db_path=db_path)
        self.llm = EnhancedOfflineLLM(model_name=llm_model)
        
        print("Enhanced OfflineRAG system initialized")
        print(f"LLM Model Info: {self.llm.get_model_info()}")
    
    def query(self, 
              question: str, 
              top_k: int = 8, 
              modality_filter: str = None,
              include_llm_response: bool = True,
              response_length: str = "medium") -> Dict[str, Any]:
        """Process a query through the enhanced RAG pipeline"""
        
        try:
            # Step 1: Retrieve relevant documents with higher top_k for better context
            search_top_k = max(top_k, 10)  # Ensure we get enough context
            search_results = self.search.search(
                query=question,
                top_k=search_top_k,
                modality_filter=modality_filter
            )
            
            # Filter results by relevance threshold
            filtered_results = [r for r in search_results if r.get('score', 0) > 0.1]
            
            response = {
                'query': question,
                'search_results': filtered_results[:top_k],  # Return requested number
                'llm_response': None,
                'citations': [],
                'confidence': 0.0,
                'metadata': {
                    'total_results_found': len(search_results),
                    'filtered_results': len(filtered_results),
                    'modality_filter': modality_filter,
                    'top_k': top_k,
                    'model_info': self.llm.get_model_info()
                }
            }
            
            # Step 2: Generate enhanced LLM response if requested
            if include_llm_response and filtered_results:
                # Determine response length
                max_length_map = {
                    "short": 200,
                    "medium": 400,
                    "long": 600
                }
                max_length = max_length_map.get(response_length, 400)
                
                # Use more results for LLM context
                llm_context_results = filtered_results[:min(8, len(filtered_results))]
                
                llm_result = self.llm.generate_rag_response(
                    query=question, 
                    search_results=llm_context_results,
                    max_length=max_length
                )
                
                response['llm_response'] = llm_result['answer']
                response['citations'] = llm_result['citations']
                response['confidence'] = llm_result['confidence']
                response['metadata']['model_used'] = llm_result.get('model_used', 'Unknown')
            
            elif include_llm_response and not filtered_results:
                response['llm_response'] = "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your query or check if the documents contain the information you're looking for."
                response['confidence'] = 0.0
            
            return response
            
        except Exception as e:
            print(f"Enhanced RAG query failed: {e}")
            return {
                'query': question,
                'search_results': [],
                'llm_response': f"I encountered an error while processing your question: {str(e)}. Please try again.",
                'citations': [],
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
    
    def cross_modal_query(self, question: str, top_k: int = 12, response_length: str = "medium") -> Dict[str, Any]:
        """Perform enhanced cross-modal query with better LLM integration"""
        
        try:
            cross_results = self.search.cross_modal_search(question, top_k=top_k)
            
            # Flatten and rank all results
            all_results = []
            for modality, results in cross_results.items():
                for result in results:
                    result['modality'] = modality  # Add modality info
                    all_results.append(result)
            
            # Sort by score and filter by relevance
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            filtered_results = [r for r in all_results if r.get('score', 0) > 0.1]
            
            # Generate enhanced LLM response using cross-modal context
            llm_response = None
            citations = []
            confidence = 0.0
            
            if filtered_results:
                # Use diverse results from different modalities for richer context
                diverse_results = self._select_diverse_results(filtered_results, max_results=10)
                
                max_length_map = {
                    "short": 250,
                    "medium": 500,
                    "long": 750
                }
                max_length = max_length_map.get(response_length, 500)
                
                llm_result = self.llm.generate_rag_response(
                    query=question, 
                    search_results=diverse_results,
                    max_length=max_length
                )
                
                llm_response = llm_result['answer']
                citations = llm_result['citations']
                confidence = llm_result['confidence']
            
            return {
                'query': question,
                'cross_modal_results': cross_results,
                'combined_results': filtered_results[:top_k],
                'llm_response': llm_response,
                'citations': citations,
                'confidence': confidence,
                'metadata': {
                    'total_results': len(all_results),
                    'filtered_results': len(filtered_results),
                    'results_by_modality': {k: len(v) for k, v in cross_results.items()},
                    'model_info': self.llm.get_model_info()
                }
            }
            
        except Exception as e:
            print(f"Enhanced cross-modal query failed: {e}")
            return {
                'query': question,
                'cross_modal_results': {},
                'combined_results': [],
                'llm_response': f"I encountered an error while processing your cross-modal query: {str(e)}",
                'citations': [],
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
    
    def _select_diverse_results(self, results: List[Dict[str, Any]], max_results: int = 10) -> List[Dict[str, Any]]:
        """Select diverse results from different modalities and sources"""
        diverse_results = []
        seen_sources = set()
        modality_counts = {'text': 0, 'image': 0, 'audio': 0, 'table': 0}
        max_per_modality = max_results // 4 + 1
        
        # First pass: ensure diversity across modalities and sources
        for result in results:
            if len(diverse_results) >= max_results:
                break
                
            modality = result.get('type', 'text')
            source = result.get('source_file', 'unknown')
            
            # Prefer results from different sources and modalities
            if (modality_counts[modality] < max_per_modality and 
                (source not in seen_sources or len(diverse_results) < max_results // 2)):
                
                diverse_results.append(result)
                seen_sources.add(source)
                modality_counts[modality] += 1
        
        # Second pass: fill remaining slots with best scores
        for result in results:
            if len(diverse_results) >= max_results:
                break
            if result not in diverse_results:
                diverse_results.append(result)
        
        return diverse_results[:max_results]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get enhanced system statistics"""
        search_stats = self.search.get_statistics()
        llm_info = self.llm.get_model_info()
        
        return {
            'search_index': search_stats,
            'llm_model': llm_info,
            'system_status': 'operational' if self.search.processor and self.llm else 'degraded',
            'capabilities': {
                'multimodal_search': bool(self.search.processor),
                'llm_generation': bool(self.llm.generator),
                'cross_modal_retrieval': True,
                'citation_support': True,
                'offline_operation': True
            }
        }
    
    def analyze_query_complexity(self, question: str) -> Dict[str, Any]:
        """Analyze query complexity to optimize retrieval strategy"""
        question_lower = question.lower()
        
        # Detect question type
        question_types = {
            'factual': ['what', 'when', 'where', 'who'],
            'analytical': ['why', 'how', 'analyze', 'compare', 'explain'],
            'procedural': ['steps', 'process', 'procedure', 'method'],
            'definitional': ['define', 'definition', 'meaning', 'concept']
        }
        
        detected_types = []
        for q_type, keywords in question_types.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_types.append(q_type)
        
        # Detect modality preferences
        modality_hints = {
            'image': ['image', 'picture', 'diagram', 'chart', 'graph', 'visual'],
            'audio': ['audio', 'sound', 'speech', 'recording', 'transcript'],
            'table': ['table', 'data', 'statistics', 'numbers', 'values']
        }
        
        preferred_modalities = []
        for modality, keywords in modality_hints.items():
            if any(keyword in question_lower for keyword in keywords):
                preferred_modalities.append(modality)
        
        # Estimate complexity
        complexity_score = len(question.split()) / 10  # Simple word count based
        if any(word in question_lower for word in ['compare', 'analyze', 'relationship', 'between']):
            complexity_score += 0.5
        
        return {
            'question_types': detected_types or ['general'],
            'preferred_modalities': preferred_modalities,
            'complexity_score': min(complexity_score, 1.0),
            'recommended_top_k': max(5, min(15, int(complexity_score * 20)))
        }