import os
from typing import List, Dict, Any, Optional
from retrieval.offline_search import OfflineSearch
from retrieval.enhanced_offline_llm import EnhancedOfflineLLM

class EnhancedOfflineRAG:
    """Enhanced offline multimodal RAG pipeline with robust LLM integration"""
    
    def __init__(self, db_path: str = "data/faiss_db", llm_model: str = "gpt2"):
        """Initialize enhanced offline RAG system"""
        print("🚀 Initializing Enhanced Offline RAG System...")
        
        # Initialize search system
        print("📊 Setting up search system...")
        self.search = OfflineSearch(db_path=db_path)
        
        # Initialize enhanced LLM
        print("🤖 Setting up enhanced LLM...")
        self.llm = EnhancedOfflineLLM(model_name=llm_model)
        
        # Test the system
        self._test_system()
        
        print("✅ Enhanced OfflineRAG system initialized successfully!")
    
    def _test_system(self):
        """Test the system components"""
        print("\n🔧 Testing system components...")
        
        # Test search
        if self.search.processor:
            stats = self.search.get_statistics()
            print(f"📊 Search Index: {stats.get('total_vectors', 0)} vectors loaded")
        else:
            print("⚠️ Search system not properly initialized")
        
        # Test LLM
        llm_test = self.llm.test_generation()
        print(f"🤖 LLM Status: {llm_test['message']}")
        
        model_info = self.llm.get_model_info()
        print(f"🔧 Model: {model_info['model_name']} ({model_info['model_type']})")
    
    def query(self, 
              question: str, 
              top_k: int = 8, 
              modality_filter: str = None,
              include_llm_response: bool = True,
              response_length: str = "medium") -> Dict[str, Any]:
        """Process a query through the enhanced RAG pipeline"""
        
        print(f"\n🔍 Processing query: '{question[:50]}...'")
        
        try:
            # Step 1: Retrieve relevant documents
            print("📊 Searching for relevant documents...")
            search_top_k = max(top_k * 2, 15)  # Get more results for better context
            search_results = self.search.search(
                query=question,
                top_k=search_top_k,
                modality_filter=modality_filter
            )
            
            print(f"📊 Found {len(search_results)} search results")
            
            # Filter results by relevance threshold
            min_score = 0.1  # Minimum relevance score
            filtered_results = [r for r in search_results if r.get('score', 0) > min_score]
            
            if not filtered_results and search_results:
                # If no results meet threshold, take top results anyway
                filtered_results = search_results[:top_k]
                print(f"⚠️ Using {len(filtered_results)} results below threshold")
            
            print(f"✅ Using {len(filtered_results)} relevant results")
            
            response = {
                'query': question,
                'search_results': filtered_results[:top_k],
                'llm_response': None,
                'citations': [],
                'confidence': 0.0,
                'metadata': {
                    'total_results_found': len(search_results),
                    'filtered_results': len(filtered_results),
                    'modality_filter': modality_filter,
                    'top_k': top_k,
                    'model_info': self.llm.get_model_info(),
                    'search_stats': self.search.get_statistics()
                }
            }
            
            # Step 2: Generate enhanced LLM response
            if include_llm_response:
                print("🤖 Generating AI response...")
                
                if filtered_results:
                    # Determine response length
                    max_length_map = {
                        "short": 200,
                        "medium": 400,
                        "long": 600
                    }
                    max_length = max_length_map.get(response_length, 400)
                    
                    # Use diverse results for better context
                    context_results = self._select_diverse_results(filtered_results, max_results=10)
                    
                    llm_result = self.llm.generate_rag_response(
                        query=question, 
                        search_results=context_results,
                        max_length=max_length
                    )
                    
                    response['llm_response'] = llm_result['answer']
                    response['citations'] = llm_result['citations']
                    response['confidence'] = llm_result['confidence']
                    response['metadata']['model_used'] = llm_result.get('model_used', 'Unknown')
                    
                    print(f"✅ Generated response with confidence: {llm_result['confidence']:.2f}")
                    
                else:
                    response['llm_response'] = self._generate_no_results_response(question)
                    response['confidence'] = 0.0
                    print("⚠️ No relevant results found")
            
            return response
            
        except Exception as e:
            print(f"❌ RAG query failed: {e}")
            return {
                'query': question,
                'search_results': [],
                'llm_response': f"I encountered an error while processing your question: {str(e)}. Please try again or rephrase your question.",
                'citations': [],
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
    
    def cross_modal_query(self, question: str, top_k: int = 12, response_length: str = "medium") -> Dict[str, Any]:
        """Perform enhanced cross-modal query with better LLM integration"""
        
        print(f"\n🔄 Processing cross-modal query: '{question[:50]}...'")
        
        try:
            cross_results = self.search.cross_modal_search(question, top_k=top_k)
            
            # Flatten and rank all results
            all_results = []
            for modality, results in cross_results.items():
                for result in results:
                    result['modality'] = modality
                    all_results.append(result)
            
            # Sort by score and filter by relevance
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            filtered_results = [r for r in all_results if r.get('score', 0) > 0.1]
            
            print(f"🔄 Cross-modal search found {len(all_results)} total results, {len(filtered_results)} relevant")
            
            # Generate enhanced LLM response using cross-modal context
            llm_response = None
            citations = []
            confidence = 0.0
            
            if filtered_results:
                print("🤖 Generating cross-modal AI response...")
                
                # Use diverse results from different modalities
                diverse_results = self._select_diverse_results(filtered_results, max_results=12)
                
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
                
                print(f"✅ Generated cross-modal response with confidence: {confidence:.2f}")
            
            return {
                'query': question,
                'cross_modal_results': cross_results,
                'combined_results': filtered_results[:top_k],
                'llm_response': llm_response or self._generate_no_results_response(question),
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
            print(f"❌ Cross-modal query failed: {e}")
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
        max_per_modality = max(2, max_results // 4)
        
        # First pass: ensure diversity across modalities and sources
        for result in results:
            if len(diverse_results) >= max_results:
                break
                
            modality = result.get('type', 'text')
            source = result.get('source_file', 'unknown')
            score = result.get('score', 0)
            
            # Prefer high-scoring results from different sources and modalities
            if (modality_counts.get(modality, 0) < max_per_modality and 
                (source not in seen_sources or len(diverse_results) < max_results // 2) and
                score > 0.05):  # Minimum score threshold
                
                diverse_results.append(result)
                seen_sources.add(source)
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        # Second pass: fill remaining slots with best scores
        for result in results:
            if len(diverse_results) >= max_results:
                break
            if result not in diverse_results and result.get('score', 0) > 0.05:
                diverse_results.append(result)
        
        print(f"📊 Selected diverse results: {dict(modality_counts)}")
        return diverse_results[:max_results]
    
    def _generate_no_results_response(self, question: str) -> str:
        """Generate a helpful response when no results are found"""
        return f"""I couldn't find any relevant information in the documents to answer your question about "{question}".

This could be because:
- The information might not be present in the uploaded documents
- The question might need to be rephrased using different keywords
- The documents might not have been processed yet

**Suggestions:**
- Try rephrasing your question with different keywords
- Check if the relevant documents have been uploaded and processed
- Make sure your question is specific and clear

Please try a different question or upload additional documents that might contain the information you're looking for."""
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get enhanced system statistics"""
        search_stats = self.search.get_statistics() if self.search.processor else {}
        llm_info = self.llm.get_model_info()
        
        return {
            'search_index': search_stats,
            'llm_model': llm_info,
            'system_status': 'operational' if self.search.processor else 'degraded',
            'capabilities': {
                'multimodal_search': bool(self.search.processor),
                'llm_generation': bool(self.llm.generator),
                'template_fallback': True,
                'cross_modal_retrieval': True,
                'citation_support': True,
                'offline_operation': True,
                'robust_error_handling': True
            },
            'performance': {
                'total_vectors': search_stats.get('total_vectors', 0),
                'model_type': llm_info.get('model_type', 'Unknown'),
                'device': llm_info.get('device', 'Unknown')
            }
        }
    
    def analyze_query_complexity(self, question: str) -> Dict[str, Any]:
        """Analyze query complexity to optimize retrieval strategy"""
        question_lower = question.lower()
        
        # Detect question type
        question_types = {
            'factual': ['what', 'when', 'where', 'who', 'which'],
            'analytical': ['why', 'how', 'analyze', 'compare', 'explain', 'describe'],
            'procedural': ['steps', 'process', 'procedure', 'method', 'how to'],
            'definitional': ['define', 'definition', 'meaning', 'concept', 'what is']
        }
        
        detected_types = []
        for q_type, keywords in question_types.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_types.append(q_type)
        
        # Detect modality preferences
        modality_hints = {
            'image': ['image', 'picture', 'diagram', 'chart', 'graph', 'visual', 'figure'],
            'audio': ['audio', 'sound', 'speech', 'recording', 'transcript', 'voice'],
            'table': ['table', 'data', 'statistics', 'numbers', 'values', 'dataset']
        }
        
        preferred_modalities = []
        for modality, keywords in modality_hints.items():
            if any(keyword in question_lower for keyword in keywords):
                preferred_modalities.append(modality)
        
        # Estimate complexity
        word_count = len(question.split())
        complexity_indicators = ['compare', 'analyze', 'relationship', 'between', 'difference', 'similarity']
        
        complexity_score = word_count / 15  # Base complexity from length
        if any(indicator in question_lower for indicator in complexity_indicators):
            complexity_score += 0.5
        if len(detected_types) > 1:
            complexity_score += 0.3
        
        complexity_score = min(complexity_score, 1.0)
        
        return {
            'question_types': detected_types or ['general'],
            'preferred_modalities': preferred_modalities,
            'complexity_score': complexity_score,
            'recommended_top_k': max(5, min(20, int(complexity_score * 25))),
            'word_count': word_count,
            'requires_cross_modal': len(preferred_modalities) > 1 or 'compare' in question_lower
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of the system"""
        health_status = {
            'overall_status': 'healthy',
            'components': {},
            'recommendations': []
        }
        
        # Check search system
        if self.search.processor:
            stats = self.search.get_statistics()
            health_status['components']['search'] = {
                'status': 'healthy',
                'vectors': stats.get('total_vectors', 0),
                'index_size_mb': stats.get('index_size_mb', 0)
            }
            
            if stats.get('total_vectors', 0) == 0:
                health_status['recommendations'].append("No documents indexed. Please upload and process documents.")
        else:
            health_status['components']['search'] = {'status': 'failed'}
            health_status['overall_status'] = 'degraded'
            health_status['recommendations'].append("Search system failed to initialize.")
        
        # Check LLM system
        llm_test = self.llm.test_generation()
        health_status['components']['llm'] = {
            'status': 'healthy' if llm_test['test_successful'] else 'degraded',
            'model': self.llm.model_name,
            'type': self.llm.get_model_info()['model_type']
        }
        
        if not llm_test['test_successful']:
            health_status['recommendations'].append("LLM generation issues detected. Using template responses.")
        
        return health_status