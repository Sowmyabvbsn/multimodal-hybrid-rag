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

        self.extractor = PDFExtractor(raw_dir=raw_dir, extracted_dir=extracted_dir)
        self.embedder = ChromaEmbeddingProcessor(
            google_api_key=google_api_key,
            db_path=db_path
        )
        self.retriever = ChromaSearch(db_path=db_path)

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

    def search(self, query: str, 
               filters: dict = None, 
               top_k: int = 5, 
               result_type: str = "semantic"):
        """
        Retrieve relevant chunks for a query.
        """
        results = self.retriever.search(
            query=query,
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