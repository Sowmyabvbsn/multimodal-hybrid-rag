import os
import torch
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings("ignore")

class EnhancedOfflineLLM:
    """Enhanced offline LLM with quantization and better RAG capabilities"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", use_quantization: bool = True):
        """Initialize enhanced offline LLM with quantization support"""
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.tokenizer = None
        self.model = None
        self.generator = None
        
        # Try to load a better model if available
        self._init_model()
    
    def _init_model(self):
        """Initialize the language model with quantization if available"""
        try:
            print(f"Loading enhanced offline LLM: {self.model_name}")
            
            # Try to use a better model for RAG
            available_models = [
                "microsoft/DialoGPT-large",
                "microsoft/DialoGPT-medium", 
                "distilgpt2",
                "gpt2"
            ]
            
            model_loaded = False
            for model_name in available_models:
                try:
                    print(f"Attempting to load {model_name}...")
                    
                    # Load tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # Configure quantization if available and requested
                    model_kwargs = {}
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
                            print(f"Quantization not available: {e}")
                    
                    # Load model
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        low_cpu_mem_usage=True,
                        **model_kwargs
                    )
                    
                    # Create text generation pipeline
                    device = 0 if torch.cuda.is_available() else -1
                    self.generator = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=device,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        max_length=1024,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1
                    )
                    
                    self.model_name = model_name
                    model_loaded = True
                    print(f"Successfully loaded {model_name}")
                    break
                    
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
                    continue
            
            if not model_loaded:
                print("Failed to load any model, falling back to template responses")
                self.generator = None
                
        except Exception as e:
            print(f"Failed to initialize enhanced LLM: {e}")
            self.generator = None
    
    def format_context_with_citations(self, search_results: List[Dict[str, Any]], max_context_length: int = 2000) -> tuple:
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
            result_text = f"[{i}] {result['text']}"
            
            # Add metadata context
            if result.get('type') == 'image' and result.get('ocr_text'):
                result_text += f" (OCR: {result['ocr_text'][:100]}...)"
            elif result.get('type') == 'audio' and result.get('transcript'):
                result_text += f" (Transcript: {result['transcript'][:100]}...)"
            
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
                'answer': "I couldn't find any relevant information to answer your question. Please try rephrasing your query or check if the documents contain the information you're looking for.",
                'citations': [],
                'confidence': 0.0
            }
        
        # Format context and citations
        context, citations = self.format_context_with_citations(search_results)
        
        if not self.generator:
            # Enhanced template-based response
            return self._enhanced_template_response(query, search_results, context, citations)
        
        try:
            # Create enhanced RAG prompt
            prompt = self._create_rag_prompt(query, context)
            
            # Generate response
            response = self.generator(
                prompt,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True,
                return_full_text=False
            )
            
            # Extract and clean generated text
            generated_text = response[0]['generated_text']
            answer = self._clean_and_enhance_response(generated_text, citations)
            
            # Calculate confidence based on search scores
            avg_score = sum(r.get('score', 0) for r in search_results) / len(search_results)
            confidence = min(avg_score * 1.2, 1.0)  # Boost confidence slightly
            
            return {
                'answer': answer,
                'citations': citations,
                'confidence': confidence,
                'model_used': self.model_name
            }
            
        except Exception as e:
            print(f"Failed to generate LLM response: {e}")
            return self._enhanced_template_response(query, search_results, context, citations)
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create an enhanced RAG prompt"""
        prompt_template = """You are a helpful AI assistant that answers questions based on provided context from documents. 

Context from retrieved documents:
{context}

Question: {query}

Instructions:
1. Answer the question using ONLY the information provided in the context
2. Be specific and cite relevant information with [number] references
3. If the context doesn't contain enough information, say so clearly
4. Provide a comprehensive but concise answer
5. Maintain accuracy and avoid speculation

Answer:"""
        
        return prompt_template.format(context=context, query=query)
    
    def _enhanced_template_response(self, query: str, search_results: List[Dict[str, Any]], context: str, citations: List[Dict]) -> Dict[str, Any]:
        """Enhanced template-based response when LLM is not available"""
        
        # Analyze query type
        query_lower = query.lower()
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        is_question = any(word in query_lower for word in question_words) or query.endswith('?')
        
        # Group results by type
        text_results = [r for r in search_results if r.get('type') == 'text']
        image_results = [r for r in search_results if r.get('type') == 'image']
        audio_results = [r for r in search_results if r.get('type') == 'audio']
        table_results = [r for r in search_results if r.get('type') == 'table']
        
        # Build response
        response_parts = []
        
        if is_question:
            response_parts.append(f"Based on the available documents, here's what I found regarding '{query}':")
        else:
            response_parts.append(f"Here are the most relevant results for '{query}':")
        
        response_parts.append("")
        
        # Add key findings from top results
        top_results = search_results[:3]
        for i, result in enumerate(top_results, 1):
            content = result['text']
            if len(content) > 150:
                content = content[:150] + "..."
            
            result_type = result.get('type', 'text').upper()
            source = result.get('source_file', 'Unknown')
            page = result.get('page_number', 'N/A')
            
            response_parts.append(f"**Key Finding {i}** [{i}]:")
            response_parts.append(f"{content}")
            response_parts.append(f"*Source: {source}, Page: {page}, Type: {result_type}*")
            response_parts.append("")
        
        # Add summary by modality
        if len(search_results) > 3:
            modality_summary = []
            if text_results:
                modality_summary.append(f"{len(text_results)} text passages")
            if image_results:
                modality_summary.append(f"{len(image_results)} images")
            if audio_results:
                modality_summary.append(f"{len(audio_results)} audio segments")
            if table_results:
                modality_summary.append(f"{len(table_results)} tables")
            
            if modality_summary:
                response_parts.append(f"**Additional Context:** Found {', '.join(modality_summary)} related to your query.")
                response_parts.append("")
        
        # Calculate confidence
        avg_score = sum(r.get('score', 0) for r in search_results) / len(search_results)
        confidence = min(avg_score, 0.8)  # Cap template confidence at 0.8
        
        return {
            'answer': "\n".join(response_parts),
            'citations': citations,
            'confidence': confidence,
            'model_used': 'Enhanced Template'
        }
    
    def _clean_and_enhance_response(self, response: str, citations: List[Dict]) -> str:
        """Clean and enhance the generated response"""
        # Remove common artifacts
        response = response.replace("<|endoftext|>", "")
        response = response.replace("</s>", "")
        response = response.replace("<pad>", "")
        
        # Split by sentences and ensure reasonable length
        sentences = response.split('. ')
        if len(sentences) > 8:
            response = '. '.join(sentences[:8]) + '.'
        
        # Ensure response ends properly
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        # Add citation summary if not present
        if '[' not in response and citations:
            response += f"\n\n*Based on {len(citations)} sources from the document collection.*"
        
        return response.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'has_generator': self.generator is not None,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'quantization_enabled': self.use_quantization and torch.cuda.is_available()
        }