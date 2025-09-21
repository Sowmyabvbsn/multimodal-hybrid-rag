import os
import torch
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
import warnings
warnings.filterwarnings("ignore")

class EnhancedOfflineLLM:
    """Enhanced offline LLM with multiple model fallbacks and better RAG capabilities"""
    
    def __init__(self, model_name: str = "gpt2", use_quantization: bool = False):
        """Initialize enhanced offline LLM with multiple fallback options"""
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.tokenizer = None
        self.model = None
        self.generator = None
        
        # Model priority list - from best to most basic
        self.model_priority = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-small", 
            "gpt2-medium",
            "gpt2",
            "distilgpt2"
        ]
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the language model with fallback options"""
        print("🤖 Initializing Enhanced Offline LLM...")
        
        # Try models in priority order
        for model_name in self.model_priority:
            try:
                print(f"📥 Attempting to load {model_name}...")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Configure model loading
                model_kwargs = {
                    "torch_dtype": torch.float32,  # Use float32 for better compatibility
                    "low_cpu_mem_usage": True
                }
                
                # Add quantization if available and requested
                if self.use_quantization and torch.cuda.is_available():
                    try:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        model_kwargs["quantization_config"] = quantization_config
                        model_kwargs["device_map"] = "auto"
                    except Exception as e:
                        print(f"⚠️ Quantization not available: {e}")
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                
                # Create text generation pipeline
                device = 0 if torch.cuda.is_available() else -1
                self.generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=device,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                self.model_name = model_name
                print(f"✅ Successfully loaded {model_name}")
                return
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
        
        # If all models fail, use template-based responses
        print("⚠️ All models failed to load, using template-based responses")
        self.generator = None
    
    def format_context_with_citations(self, search_results: List[Dict[str, Any]], max_context_length: int = 1500) -> tuple:
        """Format search results into context with proper citations"""
        context_parts = []
        citations = []
        current_length = 0
        
        for i, result in enumerate(search_results, 1):
            # Create citation
            citation = {
                'id': i,
                'source_file': result.get('source_file', 'Unknown'),
                'page_number': result.get('page_number'),
                'type': result.get('type', 'text'),
                'score': result.get('score', 0),
                'image_path': result.get('image_path'),
                'audio_path': result.get('audio_path'),
                'timestamp': result.get('timestamp')
            }
            citations.append(citation)
            
            # Format result with citation
            content = result.get('text', '')
            if len(content) > 300:
                content = content[:300] + "..."
            
            result_text = f"[{i}] {content}"
            
            # Add metadata context for multimodal content
            if result.get('type') == 'image' and result.get('ocr_text'):
                result_text += f" (Image OCR: {result['ocr_text'][:100]}...)"
            elif result.get('type') == 'audio' and result.get('transcript'):
                result_text += f" (Audio: {result['transcript'][:100]}...)"
            elif result.get('type') == 'table':
                result_text += " (Table data)"
            
            # Check length limit
            if current_length + len(result_text) > max_context_length:
                break
            
            context_parts.append(result_text)
            current_length += len(result_text)
        
        context = "\n\n".join(context_parts)
        return context, citations
    
    def generate_rag_response(self, query: str, search_results: List[Dict[str, Any]], max_length: int = 400) -> Dict[str, Any]:
        """Generate RAG response with enhanced prompting and citations"""
        
        if not search_results:
            return {
                'answer': "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your query or check if the documents contain the information you're looking for.",
                'citations': [],
                'confidence': 0.0,
                'model_used': 'No Results'
            }
        
        # Format context and citations
        context, citations = self.format_context_with_citations(search_results)
        
        if not self.generator:
            # Enhanced template-based response
            return self._enhanced_template_response(query, search_results, context, citations)
        
        try:
            # Create enhanced RAG prompt
            prompt = self._create_enhanced_rag_prompt(query, context)
            
            # Generate response with better parameters
            response = self.generator(
                prompt,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True,
                return_full_text=False
            )
            
            # Extract and clean generated text
            generated_text = response[0]['generated_text']
            answer = self._clean_and_enhance_response(generated_text, citations)
            
            # Calculate confidence based on search scores and response quality
            avg_score = sum(r.get('score', 0) for r in search_results) / len(search_results)
            response_quality = min(len(answer.split()) / 50, 1.0)  # Quality based on response length
            confidence = min((avg_score * 0.7 + response_quality * 0.3) * 1.2, 1.0)
            
            return {
                'answer': answer,
                'citations': citations,
                'confidence': confidence,
                'model_used': self.model_name
            }
            
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return self._enhanced_template_response(query, search_results, context, citations)
    
    def _create_enhanced_rag_prompt(self, query: str, context: str) -> str:
        """Create an enhanced RAG prompt with better instructions"""
        
        # Analyze query type for better prompting
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'define', 'explain']):
            instruction = "Provide a clear and comprehensive explanation"
        elif any(word in query_lower for word in ['how', 'steps', 'process']):
            instruction = "Explain the process step by step"
        elif any(word in query_lower for word in ['why', 'reason', 'cause']):
            instruction = "Explain the reasons and causes"
        elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
            instruction = "Compare and contrast the different aspects"
        else:
            instruction = "Answer comprehensively"
        
        prompt_template = f"""You are a helpful AI assistant that answers questions based on provided document context.

Context from documents:
{context}

Question: {query}

Instructions:
- {instruction} using ONLY the information from the context above
- Be specific and reference the source numbers [1], [2], etc. when citing information
- If the context doesn't contain enough information, clearly state what's missing
- Provide a well-structured and informative answer
- Keep your response focused and relevant to the question

Answer:"""
        
        return prompt_template
    
    def _enhanced_template_response(self, query: str, search_results: List[Dict[str, Any]], context: str, citations: List[Dict]) -> Dict[str, Any]:
        """Enhanced template-based response when LLM is not available"""
        
        # Analyze query type
        query_lower = query.lower()
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'define', 'explain']
        is_question = any(word in query_lower for word in question_words) or query.endswith('?')
        
        # Group results by type for better organization
        results_by_type = {'text': [], 'image': [], 'audio': [], 'table': []}
        for result in search_results:
            result_type = result.get('type', 'text')
            if result_type in results_by_type:
                results_by_type[result_type].append(result)
        
        # Build comprehensive response
        response_parts = []
        
        # Introduction
        if is_question:
            response_parts.append(f"Based on the available documents, here's what I found regarding '{query}':")
        else:
            response_parts.append(f"Here are the most relevant results for '{query}':")
        
        response_parts.append("")
        
        # Main findings from top results
        top_results = search_results[:4]  # Use top 4 results
        for i, result in enumerate(top_results, 1):
            content = result.get('text', '')
            if len(content) > 200:
                content = content[:200] + "..."
            
            result_type = result.get('type', 'text').upper()
            source = result.get('source_file', 'Unknown')
            page = result.get('page_number', 'N/A')
            score = result.get('score', 0)
            
            response_parts.append(f"**Finding {i}** [{i}] (Relevance: {score:.2f}):")
            response_parts.append(f"{content}")
            
            # Add type-specific information
            if result.get('type') == 'image' and result.get('ocr_text'):
                response_parts.append(f"*Image contains text: {result['ocr_text'][:100]}...*")
            elif result.get('type') == 'audio' and result.get('transcript'):
                response_parts.append(f"*Audio transcript: {result['transcript'][:100]}...*")
            
            response_parts.append(f"*Source: {source}, Page: {page}, Type: {result_type}*")
            response_parts.append("")
        
        # Summary by content type if multiple types found
        active_types = [t for t, results in results_by_type.items() if results]
        if len(active_types) > 1:
            type_summary = []
            for content_type in active_types:
                count = len(results_by_type[content_type])
                type_summary.append(f"{count} {content_type} {'items' if count > 1 else 'item'}")
            
            response_parts.append(f"**Content Summary:** Found {', '.join(type_summary)} related to your query.")
            response_parts.append("")
        
        # Confidence and limitations
        avg_score = sum(r.get('score', 0) for r in search_results) / len(search_results)
        if avg_score > 0.7:
            confidence_text = "high confidence"
        elif avg_score > 0.4:
            confidence_text = "moderate confidence"
        else:
            confidence_text = "low confidence"
        
        response_parts.append(f"*This response is generated with {confidence_text} based on {len(search_results)} relevant document sections.*")
        
        # Calculate confidence score
        confidence = min(avg_score * 0.8, 0.8)  # Cap template confidence at 0.8
        
        return {
            'answer': "\n".join(response_parts),
            'citations': citations,
            'confidence': confidence,
            'model_used': 'Enhanced Template System'
        }
    
    def _clean_and_enhance_response(self, response: str, citations: List[Dict]) -> str:
        """Clean and enhance the generated response"""
        # Remove common artifacts
        response = response.replace("<|endoftext|>", "")
        response = response.replace("</s>", "")
        response = response.replace("<pad>", "")
        response = response.replace("Answer:", "")
        
        # Clean up whitespace
        response = response.strip()
        
        # Split by sentences and ensure reasonable length
        sentences = response.split('. ')
        if len(sentences) > 10:
            response = '. '.join(sentences[:10]) + '.'
        
        # Ensure response ends properly
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        # Add citation summary if not present and we have citations
        if '[' not in response and citations:
            response += f"\n\n*Based on {len(citations)} sources from the document collection.*"
        
        return response.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'has_generator': self.generator is not None,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'quantization_enabled': self.use_quantization and torch.cuda.is_available(),
            'model_type': 'LLM' if self.generator else 'Template',
            'capabilities': {
                'text_generation': self.generator is not None,
                'citation_support': True,
                'multimodal_context': True,
                'fallback_responses': True
            }
        }
    
    def test_generation(self, test_query: str = "What is machine learning?") -> Dict[str, Any]:
        """Test the model's generation capabilities"""
        try:
            if not self.generator:
                return {
                    'status': 'template_only',
                    'message': 'Using template-based responses',
                    'test_successful': True
                }
            
            # Simple test generation
            test_prompt = f"Question: {test_query}\nAnswer:"
            response = self.generator(
                test_prompt,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            return {
                'status': 'llm_working',
                'message': f'LLM generation successful with {self.model_name}',
                'test_response': response[0]['generated_text'],
                'test_successful': True
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Generation test failed: {str(e)}',
                'test_successful': False
            }