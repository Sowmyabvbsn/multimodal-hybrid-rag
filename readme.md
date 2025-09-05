A comprehensive Retrieval-Augmented Generation (RAG) pipeline for multimodal PDF documents supporting text, tables, and images with advanced hybrid search capabilities.

## 🌟 Features

### Multimodal Content Processing
- **Text Extraction**: Preserves formatting, font information, and structural metadata
- **Table Processing**: Advanced table detection and structured data extraction  
- **Image Processing**: OCR integration with Tesseract and EasyOCR for embedded text
- **Metadata Preservation**: Page numbers, bounding boxes, section headers, and content types

### Advanced Search Capabilities
- **Dense Vector Search**: Semantic similarity using Sentence Transformers
- **Sparse Vector Search**: BM25 keyword matching with MiniCOIL embeddings
- **Hybrid Search**: Intelligent combination of dense and sparse results
- **Image Search**: CLIP-based visual similarity search
- **Reranking**: ColBERT late interaction for improved result relevance

### Production-Ready Features
- **Streamlit Web Interface**: User-friendly document upload and search
- **Docker Support**: Containerized deployment with all dependencies
- **Qdrant Integration**: High-performance vector database with quantization
- **Google AI Integration**: Optional Google embeddings for enhanced performance
- **Comprehensive Logging**: Detailed processing and search analytics

## 🏗️ Architecture

```
rag/
├── data/
│   ├── raw/              # PDF files to process
│   └── extracted/        # Processed chunks json file to check data preprocessing
├── ingestion/
│   ├── extract.py        # PDF content extraction using Unstructured
│   └── vector_embeddings.py  # Multi-model embedding generation
├── retrieval/
│   ├── chatbot.py  # Chat functionality
│   ├── hybrid_search.py          # Combined dense + sparse search
│   └── rag_pipeline.py          # Complete pipeline
└── streamlit_app.py         # Web interface
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Docker & Docker Compose** (for Qdrant)
3. **System Dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils
   
   # macOS
   brew install tesseract poppler
   ```

**Launch Web Interface**:
   ```bash
   streamlit run streamlit_app.py
   ```

### First Steps

1. **Upload PDFs**: Navigate to "Document Upload" and upload your PDF files
2. **Process Documents**: Click "Process Documents" and wait for completion
3. **Search**: Use "Hybrid Search" with filters to query your documents
4. **Try Examples**: Use "Example Queries" to test different search types


### Search Types Comparison

| Search Type | Best For | Technology |
|-------------|----------|------------|
| **Dense Search** | General queries, conceptual search | Dense vectors (Gemini Embedding Model) |
| **Sparse Search** | Technical terms, mixed queries | Sparse (MiniCOIL BM25) |
| **Hybrid Search** | Technical terms, mixed queries | Dense + Sparse (MiniCOIL BM25) |
| **With Reranking** | High precision requirements | + ColBERT late interaction |


