import os
from loguru import logger
from ingestion.extract import PDFExtractor
from ingestion.chroma_embeddings import ChromaEmbeddingProcessor
from retrieval.chroma_search import ChromaSearch
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self, 
                 google_api_key: str = os.getenv("GOOGLE_API_KEY"), 
                 db_path: str = "data/chroma_db",
                 raw_dir: str = "data/raw",
                 extracted_dir: str = "data/extracted"):
        
        if not google_api_key:
            raise ValueError("Google API key is required. Please set GOOGLE_API_KEY environment variable.")

        self.extractor = PDFExtractor(raw_dir=raw_dir, extracted_dir=extracted_dir)
        
        try:
            self.embedder = ChromaEmbeddingProcessor(
                google_api_key=google_api_key,
                db_path=db_path
            )
            self.retriever = ChromaSearch(db_path=db_path)
            logger.info("RAG Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {e}")
            raise

    def extract_from_pdf(self, pdf_path: str):
        """
        Extracts, chunks, summarizes, and indexes a PDF.
        """
        try:
            # Extract content
            chunks = self.extractor.extract_pdf_content(pdf_path)
            logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
            return chunks
        except Exception as e:
            logger.error(f"Failed to extract from PDF {pdf_path}: {e}")
            raise

    def embed_and_index(self, chunks: list, collection_name: str = "unified_collection"):
        """
        Process and index chunks in the vector database.
        """
        try:
            chunk_results = self.embedder.process_all_chunks(chunks)
            logger.info(f"Indexed chunks: {chunk_results}")
            return chunk_results
        except Exception as e:
            logger.error(f"Failed to embed and index chunks: {e}")
            raise

    def search(self, query: str, 
               filters: dict = None, 
               top_k: int = 5, 
               result_type: str = "semantic"):
        """
        Retrieve relevant chunks for a query.
        """
        try:
            results = self.retriever.search(
                query=query,
                filters=filters or {},
                top_k=top_k,
                result_type=result_type
            )
            
            logger.debug(f"Found {len(results)} results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


def main():
    """Example usage"""
    try:
        pipeline = RAGPipeline()
        logger.info("RAG Pipeline ready for use!")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {e}")


if __name__ == "__main__":
    main()