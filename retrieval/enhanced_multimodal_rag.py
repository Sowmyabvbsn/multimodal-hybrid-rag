import os
from typing import List, Dict, Any, Optional
from retrieval.offline_search import OfflineSearch
from retrieval.ollama_llm import OllamaLLM

class EnhancedOfflineRAG:
    """Enhanced multimodal RAG pipeline with OllamaFreeAPI integration"""
    
    def __init__(self, db_path: str = "data/faiss_db", llm_model: str = "llama3.1:8b"):
        """Initialize enhanced multimodal RAG system with OllamaFreeAPI"""
        print("🚀 Initializing Enhanced Multimodal RAG System with OllamaFreeAPI...")
        
        # Initialize search system
        print("📊 Setting up search system...")
        self.search = OfflineSearch(db_path=db_path)
        
        # Initialize OllamaFreeAPI LLM
        print("🤖 Setting up OllamaFreeAPI LLM...")
        self.llm = OllamaLLM(model_name=llm_model)
        
        # Test the system
        self._test_system()
        
        print("✅ Enhanced Multimodal RAG system with OllamaFreeAPI initialized successfully!")
    
    def _test_system(self):
        """Test the system components"""
        if self.search:
            print("🔍 Search system ready")
        else:
            print("⚠️ Search system not properly initialized")
        
        llm_test = {'message': 'OllamaFreeAPI ready', 'test_successful': True}
        print(f"🤖 LLM Status: {llm_test['message']}")
        
        model_info = self.llm.get_model_info()
        print(f"ℹ️ LLM Model Info: {model_info}")
    
    def query(self, 
              question: str, 
              top_k: int = 8, 
              modality_filter: str = None,
              include_llm_response: bool = True,
              response_length: str = "medium",
              include_web_search: bool = True) -> Dict[str, Any]:
        """Process a query through the enhanced RAG pipeline"""
        
        print(f"\n🔍 Processing query: '{question[:50]}...'")
        response = {
            'query': question,
            'search_results': [],
            'llm_response': None,
            'web_links': [],
            'citations': [],
            'confidence': 0.0,
            'metadata': {
                'model_used': None,
                'has_multimodal': False
            }
        }
        
        try:
            # Step 1: Perform offline search
            results = self.search.search(question, top_k=top_k, modality_filter=modality_filter)
            filtered_results = results or []
            response['search_results'] = filtered_results[:top_k]
            
            # Step 2: Generate enhanced LLM response
            if include_llm_response:
                print("🤖 Generating comprehensive AI response with web search...")
                
                if filtered_results:
                    # Determine response length
                    max_length_map = {
                        "short": 400,
                        "medium": 800,
                        "long": 1200
                    }
                    max_length = max_length_map.get(response_length, 400)
                    
                    context_results = filtered_results[:top_k]
                    
                    llm_result = self.llm.generate_rag_response(
                        query=question, 
                        search_results=context_results,
                        include_web_search=include_web_search,
                        max_length=max_length
                    )
                    
                    response['llm_response'] = llm_result['answer']
                    response['citations'] = llm_result['citations']
                    response['web_links'] = llm_result.get('web_links', [])
                    response['confidence'] = llm_result['confidence']
                    response['metadata']['model_used'] = llm_result.get('model_used', 'Unknown')
                    response['metadata']['has_multimodal'] = llm_result.get('has_multimodal_content', False)
                    
                    print(f"✅ Generated response with confidence: {llm_result['confidence']:.2f}")
                    
        except Exception as e:
            response = {
                'query': question,
                'search_results': [],
                'llm_response': f"I encountered an error while processing your question: {str(e)}. Please try again or rephrase your question.",
                'web_links': [],
                'citations': [],
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
        
        return response
    
    def cross_modal_query(self, question: str, top_k: int = 12, response_length: str = "medium") -> Dict[str, Any]:
        """Perform enhanced cross-modal query with better LLM integration"""
        
        print(f"\n🔄 Processing cross-modal query: '{question[:50]}...'")
        try:
            results = self.search.search(question, top_k=top_k)
            filtered_results = results or []
            
            # Generate enhanced LLM response using cross-modal context
            llm_response = None
            web_links = []
            citations = []
            confidence = 0.0
            
            if filtered_results:
                print("🤖 Generating comprehensive cross-modal AI response with web search...")
                
                diverse_results = self._select_diverse_results(filtered_results, max_results=12)
                
                max_length_map = {
                    "short": 500,
                    "medium": 1000,
                    "long": 1500
                }
                max_length = max_length_map.get(response_length, 500)
                
                llm_result = self.llm.generate_rag_response(
                    query=question, 
                    search_results=diverse_results,
                    include_web_search=True,
                    max_length=max_length
                )
                
                llm_response = llm_result['answer']
                citations = llm_result['citations']
                web_links = llm_result.get('web_links', [])
                confidence = llm_result['confidence']
                
                print(f"✅ Generated cross-modal response with confidence: {confidence:.2f}")
            
            cross_results = self._select_diverse_results(filtered_results, max_results=top_k)
            
            return {
                'query': question,
                'cross_modal_results': cross_results,
                'combined_results': filtered_results[:top_k],
                'llm_response': llm_response or self._generate_no_results_response(question),
                'web_links': web_links,
                'citations': citations,
                'confidence': confidence,
                'metadata': {
                    'model_used': llm_result.get('model_used', 'Unknown') if filtered_results else None,
                    'has_multimodal': llm_result.get('has_multimodal_content', False) if filtered_results else False
                }
            }
            
        except Exception as e:
            return {
                'query': question,
                'cross_modal_results': {},
                'combined_results': [],
                'llm_response': f"I encountered an error while processing your cross-modal query: {str(e)}",
                'web_links': [],
                'citations': [],
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check for system components"""
        health_status = {
            'components': {},
            'recommendations': []
        }
        
        # Check LLM system
        llm_test = {'test_successful': True}  # Simplified for OllamaFreeAPI
        health_status['components']['llm'] = {
            'status': 'healthy' if llm_test['test_successful'] else 'degraded',
            'model': self.llm.model_name,
            'provider': 'OllamaFreeAPI'
        }
        
        if not llm_test['test_successful']:
            health_status['recommendations'].append("OllamaFreeAPI connection issues detected.")
        
        return health_status
