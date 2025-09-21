import os
import json
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class OfflineLLM:
    """Offline LLM for RAG responses using Hugging Face transformers"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize offline LLM"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.generator = None
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the language model"""
        try:
            logger.info(f"Loading offline LLM: {self.model_name}")
            
            # For better offline RAG, use a more suitable model
            # You can replace with other models like:
            # - "microsoft/DialoGPT-small" (faster, smaller)
            # - "gpt2" (classic, reliable)
            # - "distilgpt2" (faster version of GPT-2)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings for CPU/GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            logger.info(f"Successfully loaded {self.model_name} on {device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            logger.info("Falling back to simple template-based responses")
            self.generator = None
    
    def format_context(self, search_results: List[Dict[str, Any]], max_context_length: int = 2000) -> str:
        """Format search results into context for LLM"""
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(search_results, 1):
            # Format result with citation
            result_text = f"[{i}] {result['text']}"
            
            # Add source information
            source_info = f" (Source: {result['source_file']}"
            if result.get('page_number'):
                source_info += f", Page: {result['page_number']}"
            if result.get('timestamp'):
                source_info += f", Time: {result['timestamp']:.1f}s"
            source_info += ")"
            
            full_result = result_text + source_info
            
            # Check if adding this result would exceed max length
            if current_length + len(full_result) > max_context_length:
                break
            
            context_parts.append(full_result)
            current_length += len(full_result)
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, query: str, search_results: List[Dict[str, Any]], max_length: int = 300) -> str:
        """Generate response using retrieved context"""
        
        if not search_results:
            return "I couldn't find any relevant information to answer your question."
        
        # Format context
        context = self.format_context(search_results)
        
        if not self.generator:
            # Fallback to template-based response
            return self._template_response(query, search_results, context)
        
        try:
            # Create prompt for the LLM
            prompt = f"""Based on the following information, please answer the question concisely and accurately.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response
            response = self.generator(
                prompt,
                max_length=len(prompt.split()) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Extract only the answer part
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            # Clean up the response
            answer = self._clean_response(answer)
            
            # Add citations
            answer_with_citations = self._add_citations(answer, search_results)
            
            return answer_with_citations
            
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            return self._template_response(query, search_results, context)
    
    def _template_response(self, query: str, search_results: List[Dict[str, Any]], context: str) -> str:
        """Fallback template-based response"""
        
        # Simple extractive approach
        response_parts = []
        response_parts.append(f"Based on the available information, here's what I found regarding '{query}':")
        response_parts.append("")
        
        # Add top results with citations
        for i, result in enumerate(search_results[:3], 1):
            content = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            response_parts.append(f"[{i}] {content}")
            response_parts.append("")
        
        # Add source summary
        sources = list(set(result['source_file'] for result in search_results[:3]))
        response_parts.append(f"Sources: {', '.join(sources)}")
        
        return "\n".join(response_parts)
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response"""
        # Remove common artifacts
        response = response.replace("<|endoftext|>", "")
        response = response.replace("</s>", "")
        
        # Split by sentences and take reasonable length
        sentences = response.split('. ')
        if len(sentences) > 5:
            response = '. '.join(sentences[:5]) + '.'
        
        return response.strip()
    
    def _add_citations(self, response: str, search_results: List[Dict[str, Any]]) -> str:
        """Add citation numbers to response"""
        # Simple approach: add citations at the end
        citations = []
        for i, result in enumerate(search_results[:5], 1):
            citation = f"[{i}] {result['source_file']}"
            if result.get('page_number'):
                citation += f", Page {result['page_number']}"
            if result.get('timestamp'):
                citation += f", {result['timestamp']:.1f}s"
            citations.append(citation)
        
        if citations:
            response += "\n\nSources:\n" + "\n".join(citations)
        
        return response
    
    def chat_response(self, query: str, search_results: List[Dict[str, Any]], conversation_history: List[str] = None) -> str:
        """Generate conversational response with context"""
        
        # For now, use the same generation method
        # Could be enhanced with conversation history
        return self.generate_response(query, search_results)

def main():
    """Test offline LLM"""
    llm = OfflineLLM()
    
    # Mock search results for testing
    mock_results = [
        {
            'text': 'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
            'source_file': 'ml_basics.pdf',
            'page_number': 1,
            'type': 'text'
        },
        {
            'text': 'Neural networks are computing systems inspired by biological neural networks.',
            'source_file': 'neural_nets.pdf',
            'page_number': 3,
            'type': 'text'
        }
    ]
    
    response = llm.generate_response("What is machine learning?", mock_results)
    print("Generated Response:")
    print(response)

if __name__ == "__main__":
    main()