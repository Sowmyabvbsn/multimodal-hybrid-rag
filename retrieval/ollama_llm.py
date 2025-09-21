import os
import json
import requests
from typing import List, Dict, Any, Optional
from ollamafreeapi import OllamaAPI
import time
import re
from duckduckgo_search import DDGS

class OllamaLLM:
    """Enhanced LLM using OllamaFreeAPI with web search capabilities"""
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        """Initialize OllamaFreeAPI client"""
        self.model_name = model_name
        self.client = OllamaAPI()
        self.available_models = [
            "llama3.1:8b", "llama3.1:70b", "llama3.1:405b",
            "llama3:8b", "llama3:70b", "gemma2:9b", "gemma2:27b",
            "mistral:7b", "mixtral:8x7b", "codellama:7b", "codellama:13b",
            "phi3:3.8b", "phi3:14b", "qwen2:7b", "qwen2:72b"
        ]
        
        # Test connection
        self._test_connection()
        
        print(f"✅ OllamaLLM initialized with model: {self.model_name}")
    
    def _test_connection(self):
        """Test if the API is working"""
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello, are you working?"}],
                stream=False
            )
            print("🔗 OllamaFreeAPI connection successful!")
        except Exception as e:
            print(f"⚠️ OllamaFreeAPI connection issue: {e}")
            # Fallback to a different model
            if self.model_name != "llama3:8b":
                self.model_name = "llama3:8b"
                print(f"🔄 Switching to fallback model: {self.model_name}")
    
    def search_web(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Search the web for additional context"""
        try:
            with DDGS() as ddgs:
                results = []
                for result in ddgs.text(query, max_results=max_results):
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('href', ''),
                        'snippet': result.get('body', '')
                    })
                return results
        except Exception as e:
            print(f"Web search failed: {e}")
            return []
    
    def format_context_with_citations(self, search_results: List[Dict[str, Any]], web_results: List[Dict[str, str]] = None) -> tuple:
        """Format search results and web results into context with citations"""
        context_parts = []
        citations = []
        current_length = 0
        max_context_length = 3000
        
        # Add document results
        for i, result in enumerate(search_results, 1):
            citation = {
                'id': i,
                'source_file': result.get('source_file', 'Unknown'),
                'page_number': result.get('page_number'),
                'type': result.get('type', 'text'),
                'score': result.get('score', 0),
                'source_type': 'document'
            }
            citations.append(citation)
            
            content = result.get('text', '')
            if len(content) > 300:
                content = content[:300] + "..."
            
            result_text = f"[{i}] {content}"
            
            # Add multimodal context
            if result.get('type') == 'image' and result.get('ocr_text'):
                result_text += f" (Image OCR: {result['ocr_text'][:100]}...)"
            elif result.get('type') == 'audio' and result.get('transcript'):
                result_text += f" (Audio transcript: {result['transcript'][:100]}...)"
            elif result.get('type') == 'table':
                result_text += " (Table data)"
            
            if current_length + len(result_text) > max_context_length:
                break
            
            context_parts.append(result_text)
            current_length += len(result_text)
        
        # Add web results
        if web_results:
            web_start_id = len(citations) + 1
            for i, web_result in enumerate(web_results, web_start_id):
                if current_length > max_context_length:
                    break
                
                citation = {
                    'id': i,
                    'title': web_result.get('title', 'Web Source'),
                    'url': web_result.get('url', ''),
                    'source_type': 'web'
                }
                citations.append(citation)
                
                web_text = f"[{i}] {web_result.get('snippet', '')[:200]}..."
                if current_length + len(web_text) < max_context_length:
                    context_parts.append(web_text)
                    current_length += len(web_text)
        
        context = "\n\n".join(context_parts)
        return context, citations
    
    def generate_rag_response(self, query: str, search_results: List[Dict[str, Any]], 
                            include_web_search: bool = True, max_length: int = 800) -> Dict[str, Any]:
        """Generate comprehensive RAG response with web search integration"""
        
        if not search_results and not include_web_search:
            return {
                'answer': "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your query or upload more relevant documents.",
                'citations': [],
                'web_links': [],
                'confidence': 0.0,
                'model_used': self.model_name
            }
        
        # Perform web search for additional context
        web_results = []
        if include_web_search:
            print("🌐 Searching web for additional context...")
            web_results = self.search_web(query, max_results=3)
        
        # Format context and citations
        context, citations = self.format_context_with_citations(search_results, web_results)
        
        try:
            # Create enhanced RAG prompt
            prompt = self._create_comprehensive_prompt(query, context, search_results, web_results)
            
            # Generate response using OllamaFreeAPI
            print(f"🤖 Generating response with {self.model_name}...")
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            
            # Extract the response text
            if isinstance(response, dict) and 'message' in response:
                answer = response['message'].get('content', '')
            elif isinstance(response, str):
                answer = response
            else:
                answer = str(response)
            
            # Clean and enhance the response
            answer = self._clean_and_enhance_response(answer, citations)
            
            # Calculate confidence
            doc_score = sum(r.get('score', 0) for r in search_results) / max(len(search_results), 1)
            web_bonus = 0.1 if web_results else 0
            confidence = min(doc_score + web_bonus, 1.0)
            
            # Extract web links
            web_links = [{'title': wr.get('title', ''), 'url': wr.get('url', '')} 
                        for wr in web_results if wr.get('url')]
            
            return {
                'answer': answer,
                'citations': citations,
                'web_links': web_links,
                'confidence': confidence,
                'model_used': self.model_name,
                'has_multimodal_content': any(r.get('type') in ['image', 'audio', 'table'] for r in search_results)
            }
            
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return self._fallback_response(query, search_results, citations, web_results)
    
    def _create_comprehensive_prompt(self, query: str, context: str, 
                                   search_results: List[Dict], web_results: List[Dict]) -> str:
        """Create a comprehensive prompt for better responses"""
        
        # Analyze content types
        content_types = set(r.get('type', 'text') for r in search_results)
        multimodal_note = ""
        if 'image' in content_types:
            multimodal_note += "- Images with visual content and OCR text\n"
        if 'audio' in content_types:
            multimodal_note += "- Audio files with transcripts\n"
        if 'table' in content_types:
            multimodal_note += "- Tables with structured data\n"
        
        web_note = f"\n\nAdditional web sources (for broader context):\n{len(web_results)} recent web sources included" if web_results else ""
        
        prompt = f"""You are an expert AI assistant providing comprehensive answers based on multimodal document analysis and web research.

CONTEXT FROM DOCUMENTS:
{context}
{web_note}

MULTIMODAL CONTENT AVAILABLE:
{multimodal_note if multimodal_note else "- Text documents"}

USER QUESTION: {query}

INSTRUCTIONS:
1. Provide a comprehensive, well-structured answer using the document context as the PRIMARY source
2. Reference specific citations using [number] format when mentioning information
3. If multimodal content (images, audio, tables) is relevant, explicitly mention insights from them
4. Use web sources to provide additional context or recent information, but clearly distinguish them
5. Structure your response with clear sections if the topic is complex
6. Include practical implications or applications when relevant
7. If the question cannot be fully answered, clearly state what information is missing
8. Maintain accuracy and avoid speculation beyond the provided context

RESPONSE FORMAT:
- Start with a direct answer to the question
- Provide detailed explanation with citations
- Include relevant insights from multimodal content
- End with additional resources or next steps if applicable

Generate a comprehensive, accurate, and helpful response:"""
        
        return prompt
    
    def _clean_and_enhance_response(self, response: str, citations: List[Dict]) -> str:
        """Clean and enhance the generated response"""
        # Remove common artifacts
        response = response.strip()
        
        # Ensure proper citation format
        response = re.sub(r'\[(\d+)\]', r'[\1]', response)
        
        # Add citation summary if not present
        if citations and '[' not in response:
            doc_citations = [c for c in citations if c.get('source_type') == 'document']
            web_citations = [c for c in citations if c.get('source_type') == 'web']
            
            citation_summary = f"\n\n**Sources:**"
            if doc_citations:
                citation_summary += f"\n- Document sources: {len(doc_citations)} references"
            if web_citations:
                citation_summary += f"\n- Web sources: {len(web_citations)} additional references"
            
            response += citation_summary
        
        return response
    
    def _fallback_response(self, query: str, search_results: List[Dict], 
                          citations: List[Dict], web_results: List[Dict]) -> Dict[str, Any]:
        """Fallback response when LLM fails"""
        
        response_parts = []
        response_parts.append(f"Based on the available information, here's what I found regarding '{query}':")
        response_parts.append("")
        
        # Add top document results
        for i, result in enumerate(search_results[:3], 1):
            content = result.get('text', '')[:200] + "..."
            response_parts.append(f"**Finding {i}** [{i}]:")
            response_parts.append(content)
            response_parts.append("")
        
        # Add web results
        if web_results:
            response_parts.append("**Additional Web Sources:**")
            for i, web_result in enumerate(web_results, len(search_results) + 1):
                response_parts.append(f"[{i}] {web_result.get('title', 'Web Source')}")
                response_parts.append(f"   {web_result.get('snippet', '')[:150]}...")
                response_parts.append("")
        
        web_links = [{'title': wr.get('title', ''), 'url': wr.get('url', '')} 
                    for wr in web_results if wr.get('url')]
        
        return {
            'answer': "\n".join(response_parts),
            'citations': citations,
            'web_links': web_links,
            'confidence': 0.6,
            'model_used': 'Fallback Template',
            'has_multimodal_content': any(r.get('type') in ['image', 'audio', 'table'] for r in search_results)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'available_models': self.available_models,
            'capabilities': {
                'text_generation': True,
                'web_search': True,
                'multimodal_context': True,
                'citation_support': True,
                'external_links': True
            },
            'provider': 'OllamaFreeAPI'
        }
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        if model_name in self.available_models:
            self.model_name = model_name
            self._test_connection()
            return True
        return False