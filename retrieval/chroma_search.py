import os
from typing import List, Dict, Any
from loguru import logger
from ingestion.chroma_embeddings import ChromaEmbeddingProcessor
from dotenv import load_dotenv

load_dotenv()


class ChromaSearch:
    def __init__(self, db_path: str = "data/chroma_db"):
        """Initialize ChromaDB search"""
        try:
            self.processor = ChromaEmbeddingProcessor(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                db_path=db_path
            )
            
            if self.processor.test_connection():
                logger.info("ChromaSearch initialized successfully")
            else:
                logger.error("ChromaSearch initialization failed: No ChromaDB connection")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaSearch: {e}")
            self.processor = None
    
    def search(self, query: str, filters: Dict = None, top_k: int = 10, result_type: str = "semantic") -> List[Dict[str, Any]]:
        """Perform search and return ranked results"""
        
        if not self.processor:
            logger.error("Cannot perform search: ChromaDB not initialized")
            return []
        
        try:
            results = self.processor.search(
                query=query,
                filters=filters,
                top_k=top_k
            )
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def process_results(self, results: List[Dict]) -> List[Dict]:
        """Process search results (already formatted by ChromaEmbeddingProcessor)"""
        return results


def main():
    # Test search
    search = ChromaSearch()
    
    if search.processor:
        results = search.search("power failure troubleshooting", top_k=5)
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Score: {result['score']:.4f}")
            print(f"Type: {result['type']}")
            print(f"Source: {result['source_file']}")
            print(f"Page: {result['page_number']}")
            print(f"Text: {result['text'][:200]}...")


if __name__ == "__main__":
    main()