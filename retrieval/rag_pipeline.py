import os

from loguru import logger
from ingestion.extract import PDFExtractor
from ingestion.vector_embeddings import QdrantEmbeddingProcessor
from retrieval.hybrid_search import HybridSearch
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self, 
                 google_api_key: str = os.getenv("GOOGLE_API_KEY"), 
                 qdrant_url: str = os.getenv("QDRANT_URL"),
                 qdrant_api_key: str = os.getenv("QDRANT_API_KEY"),
                 raw_dir: str = "data/raw",
                 extracted_dir: str = "data/extracted"):

        self.extractor = PDFExtractor(raw_dir=raw_dir, extracted_dir=extracted_dir)
        self.embedder = QdrantEmbeddingProcessor(
            google_api_key=google_api_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        self.retriever = HybridSearch(collection_name="unified_collection")

    def extract_from_pdf(self, pdf_path: str):
        """
        Extracts, chunks, summarizes, and indexes a PDF.
        """
        # Extract content
        chunks = self.extractor.extract_pdf_content(pdf_path)

        return chunks

    def embed_and_index(self, chunks: list, collection_name: str = "unified_collection"):

        chunk_results = self.embedder.process_all_chunks(chunks)

        return chunk_results

    def hybrid_search(self, query: str, 
               collection_name: str = "unified_collection", 
               filters: dict = None, 
               top_k: int = 5, 
               result_type: str = "hybrid_dense"):
        """
        Retrieve relevant chunks for a query.
        """
        results = self.retriever.search(
            query=query,
            collection_name=collection_name,
            filters=filters or {},
            top_k=top_k,
            result_type=result_type
        )
        
        logger.debug(f"Processed {len(results)} results.")

        return self.retriever.process_results(results)

# # Example usage
# if __name__ == "__main__":
#     pipeline = RAGPipeline()

    # # Step 1: Upload and index a PDF
    # pdf_path = "data/raw/example.pdf"
    # chunks = pipeline.extract_from_pdf(pdf_path)
    # chunk_results = pipeline.embed_and_index(chunks)

    # # Step 2: Query
    # query = "What are the main findings in the report?"
    # results = pipeline.search(query, top_k=3)
    # for r in results:
    #     print(f"Score: {r['score']:.2f} | Page: {r['page_number']} | Type: {r['type']}")
    #     print(f"Text: {r['text'][:200]}...\n")