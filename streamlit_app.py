import streamlit as st
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from PIL import Image
import tempfile
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our RAG components
try:
    from retrieval.rag_pipeline import RAGPipeline
except ImportError as e:
    st.error(f"Failed to import RAG components: {e}")
    st.stop()


# Configure Streamlit page
st.set_page_config(
    page_title="PDF RAG Pipeline",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .score-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .metadata-info {
        color: #6c757d;
        font-size: 0.9rem;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'source_files' not in st.session_state:
        st.session_state.source_files = []
    

def setup_directories():
    """Create necessary directories"""
    directories = ["data/raw", "data/extracted"]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_source_files():
    """Load all file names from data/raw into session state"""
    raw_dir = Path("data/raw")    
    files = [f.name for f in raw_dir.glob("*") if f.is_file()]
    
    st.session_state.source_files = files
    return files

def initialize_pipeline():
    """Initialize the RAG pipeline"""
    if st.session_state.pipeline is not None:
        return st.session_state.pipeline
    
    try:
        pipeline = RAGPipeline()
        
        # Check if pipeline initialization was successful
        if not hasattr(pipeline, 'embedder') or pipeline.embedder.chroma_client is None:
            st.error("❌ Failed to connect to ChromaDB database. Please check your setup.")
            st.info("💡 ChromaDB will be created locally in the data/chroma_db directory")
            return None
        
        load_source_files()
        st.session_state.pipeline = pipeline
        st.session_state.retriever = pipeline.retriever
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return None

def render_search_result(result, index: int, search_type: str = "hybrid_dense"):
    """Render a single search result"""
    with st.container():
        st.markdown(f"""
            <div class="search-result" style="padding: 0.5rem 1rem; margin-bottom: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span class="score-badge" style="font-weight: bold; background: #e3f0ff; color: #1769aa; border-radius: 4px; padding: 0.2rem 0.5rem; font-size: 0.9rem;">
                        Score: {result.get('score', 0):.4f}
                    </span>
                    <h4 style="margin: 0; font-size: 1.1rem;">Result {index + 1}</h4>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Content preview
        content = result.get('text', '')
        if len(content) > 300:
            with st.expander("Show full content", expanded=False):
                st.text(content)
            st.write(f"**Preview:** {content[:300]}...")
        else:
            st.write(f"**Content:** {content}")
        
        # Metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**Type:** {result.get('type', 'unknown')}")
        with col2:
            st.write(f"**Source:** {result.get('source_file', 'N/A')}")
        with col3:
            st.write(f"**Page:** {result.get('page_number', 'N/A')}")
        with col4:
            if result.get('section_header'):
                st.write(f"**Section:** {result.get('section_header', 'N/A')}")

        # Show image if available
        if result.get('image_path') and os.path.exists(result['image_path']):
            try:
                with st.expander("View Image", expanded=False):
                    image = Image.open(result['image_path'])
                    st.image(image, caption=f"Image from page {result.get('page_number', 'N/A')}", width=300)
            except Exception as e:
                st.write(f"Error loading image: {e}")
        
        # Render Table html
        if result.get('type') == 'table' and result.get('content'):
            try:
                with st.expander("View Table", expanded=False):
                    st.markdown(result['content'], unsafe_allow_html=True)
            except Exception as e:
                st.write(f"Error rendering table: {e}")
        
        st.markdown("---")

def document_upload_page():
    """Document upload and ingestion page"""
    st.markdown('<h1 class="main-header">📚 Document Upload & Processing</h1>', unsafe_allow_html=True)
    
    # File uploader
    st.subheader("Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to process"
    )
    
    if uploaded_files:
        saved_files = []
        st.write(f"📄 {len(uploaded_files)} file(s) selected:")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size:,} bytes)")


        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            process_btn = st.button("🚀 Process Documents", type="primary")
        
        if process_btn:
            # Save uploaded files
            setup_directories()
            
            with st.spinner("Saving uploaded files..."):
                saved_files = []
                for file in uploaded_files:
                    file_path = Path("data/raw") / file.name
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    saved_files.append(file_path)
                
                st.success(f"✅ Saved {len(saved_files)} files")
            
            # Process documents
            with st.spinner("Processing documents... This may take a few minutes."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Create collections
                    status_text.text("Creating collections...")
                    progress_bar.progress(10)
                    
                    # Process PDFs
                    status_text.text("Extracting and processing PDFs...")
                    progress_bar.progress(30)
                    
                    text_chunks = 0
                    table_chunks = 0
                    image_chunks = 0
                    ingestion_result = {}

                    for file in saved_files:
                        status_text.text(f"Processing {file}...")
                        chunks = st.session_state.pipeline.extract_from_pdf(file)
                        status_text.text(f"Indexing {file}...")
                        chunk_results = st.session_state.pipeline.embed_and_index(chunks)
                        ingestion_result[str(file)] = chunk_results
                        text_chunks += chunk_results.get("text_chunks", 0)
                        table_chunks += chunk_results.get("table_chunks", 0)
                        image_chunks += chunk_results.get("image_chunks", 0)

                    progress_bar.progress(80)
                    
                    total_chunks = text_chunks + table_chunks + image_chunks
                    if total_chunks > 0:
                        
                        progress_bar.progress(100)
                        status_text.text("Processing completed!")
                        
                        st.session_state.documents_processed = True
                        
                        # Show results
                        st.markdown('<div class="success-message">', unsafe_allow_html=True)
                        st.success("🎉 Documents processed successfully!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Chunks", total_chunks)
                        with col2:
                            st.metric("PDFs Processed", len(saved_files))
                        
                        # Show per-PDF statistics
                        if ingestion_result:
                            st.subheader("📊 Processing Statistics")
                            pdf_stats = []
                            for pdf_name, stats in ingestion_result.items():
                                pdf_stats.append({
                                    "PDF": Path(pdf_name).name,
                                    "Total Chunks": stats["text_chunks"]+ stats["table_chunks"] + stats["image_chunks"],
                                    "Text": stats["text_chunks"],
                                    "Tables": stats["table_chunks"],
                                    "Images": stats["image_chunks"]
                                })
                            
                            df = pd.DataFrame(pdf_stats)
                            st.dataframe(df, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("Processing failed: No chunks were created")
                
                except Exception as e:
                    st.error(f"Error during processing: {e}")
                finally:
                    progress_bar.empty()
                    status_text.empty()
    
    # Show current status
    if st.session_state.documents_processed:
        st.info("✅ Documents are ready for search. Go to the Search pages to query your documents.")
    else:
        st.info("ℹ️ Upload and process PDF documents to enable search functionality.")


def semantic_search_page():
    """Semantic search page"""
    if not st.session_state.retriever and st.session_state.pipeline:
        st.session_state.retriever = st.session_state.pipeline.retriever

    st.markdown('<h1 class="main-header">🔍 Semantic Search</h1>', unsafe_allow_html=True)

    if not st.session_state.documents_processed:
        st.warning("⚠️ Please upload and process documents first before searching.")
        return

    st.subheader("Smart Search: Semantic similarity matching")

    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., sentence embeddings, transformer models, NLP techniques",
            help="Semantic search works best with descriptive queries"
        )
    
    with col2:
        top_k = st.number_input("Results", min_value=1, max_value=20, value=5)
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        content_type = st.selectbox(
            "Content Type",
            options=["All", "text", "table", "image"],
            help="Filter by content type"
        )
    with col2:
        source_file = st.selectbox(
            "Source File (optional)",
            options=["All Files"] + st.session_state.source_files,
            help="Filter by specific source file"
        )
    with col3:
        page_number = st.number_input(
            "Page Number (optional)",
            min_value=0,
            value=0,
            help="Filter by specific page (0 = all pages)"
        )
    
    # Search button
    if st.button("🚀 Semantic Search", type="primary") and query and st.session_state.retriever:
        with st.spinner("Performing search..."):
            try:
                # Apply filters
                type_filter = None if content_type == "All" else content_type
                page_filter = None if page_number == 0 else page_number
                source_filter = None if source_file == "All Files" else source_file

                filters = {}
                if type_filter:
                    filters["type"] = type_filter
                if page_filter:
                    filters["page_number"] = page_filter
                if source_filter:
                    filters["source_file"] = source_filter

                # Perform search
                results = st.session_state.retriever.search(
                    query=query,
                    filters=filters,
                    top_k=top_k
                )

                # Add to search history
                st.session_state.search_history.append({
                    "query": query,
                    "type": "semantic",
                    "results_count": len(results),
                    "timestamp": time.time()
                })
                
                # Display results
                if results:
                    st.subheader(f"🎯 Found {len(results)} results")
                    
                    for i, result in enumerate(results):
                        render_search_result(result, i, "semantic")
                else:
                    st.info("No results found. Try a different query or adjust filters.")
                    
            except Exception as e:
                st.error(f"Semantic search failed: {e}")
                st.exception(e)

def chatbot_page():
    """Chatbot page for conversational search"""
    st.markdown('<h1 class="main-header">💬 AI Chatbot</h1>', unsafe_allow_html=True)
    
    if not st.session_state.documents_processed:
        st.warning("⚠️ Please upload and process documents first before using the chatbot.")
        return
    
    st.info("🚧 Chatbot functionality coming soon! For now, please use the Semantic Search page.")

def analytics_page():
    """Analytics and statistics page"""
    st.markdown('<h1 class="main-header">📈 Analytics</h1>', unsafe_allow_html=True)
    
    if not st.session_state.documents_processed:
        st.warning("⚠️ No data to analyze. Please upload and process documents first.")
        return
    
    # Search history
    if st.session_state.search_history:
        st.subheader("🔍 Search History")
        
        history_df = pd.DataFrame(st.session_state.search_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], unit='s')
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        st.dataframe(history_df, use_container_width=True)
        
        # Search statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Searches", len(history_df))
        with col2:
            avg_results = history_df['results_count'].mean() if len(history_df) > 0 else 0
            st.metric("Avg Results per Search", f"{avg_results:.1f}")
        with col3:
            recent_searches = len(history_df[history_df['timestamp'] > pd.Timestamp.now() - pd.Timedelta(hours=24)])
            st.metric("Searches (24h)", recent_searches)
    else:
        st.info("No search history available yet. Perform some searches to see analytics.")
    
    # Document statistics
    if st.session_state.pipeline and st.session_state.pipeline.embedder:
        st.subheader("📊 Document Statistics")
        try:
            # Get collection info
            collection = st.session_state.pipeline.embedder.collection
            count = collection.count()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", count)
            with col2:
                st.metric("Source Files", len(st.session_state.source_files))
                
        except Exception as e:
            st.error(f"Failed to get document statistics: {e}")

def main():
    """Main Streamlit application"""
    initialize_session_state()
    setup_directories()
    
    # Initialize pipeline
    pipeline = initialize_pipeline()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("📚 PDF RAG Pipeline")
        st.markdown("---")
        
        page = st.selectbox(
            "Choose a page:",
            [
                "📤 Document Upload",
                "🔍 Semantic Search",
                "💬 Chatbot",
                "📈 Analytics"
            ]
        )
        
        st.markdown("---")
        st.subheader("System Status")
        
        # Add connection test button
        if st.button("🔄 Test Connection"):
            if pipeline and hasattr(pipeline, 'embedder') and pipeline.embedder:
                with st.spinner("Testing connection..."):
                    if pipeline.embedder.test_connection():
                        st.success("✅ Connection test passed!")
                    else:
                        st.error("❌ Connection test failed!")
            else:
                st.error("❌ Pipeline not initialized")
        
        # Connection status
        try:
            if pipeline and hasattr(pipeline, 'embedder'):
                if pipeline.embedder and pipeline.embedder.chroma_client:
                    st.success("✅ Pipeline Ready")
                else:
                    st.error("❌ ChromaDB Connection Failed")
            else:
                st.warning("⚠️ Pipeline Not Initialized")
        except Exception:
            st.error("❌ Connection Error")
        
        if st.session_state.documents_processed:
            st.success("✅ Documents Processed")
        else:
            st.info("ℹ️ No Documents")
        
        st.markdown("---")
        st.markdown("""
        ### About
        This is a complete multimodal RAG pipeline supporting:
        - **Text, Tables, and Images** from PDFs
        - **Semantic search** with sentence transformers
        - **Local ChromaDB** vector storage
        - **Easy setup** with no external dependencies
        """)
        
        st.markdown("---")
        st.markdown("Built with Streamlit, ChromaDB, and Sentence Transformers")
    
    # Show connection troubleshooting if there are issues
    if pipeline and hasattr(pipeline, 'embedder') and pipeline.embedder and not pipeline.embedder.chroma_client:
        st.error("🚨 ChromaDB Connection Failed")
        st.markdown("""
        ### Troubleshooting Steps:
        1. **Check disk space** - ChromaDB needs space to store the database
        2. **Check file permissions** - Make sure the app can write to the data directory
        3. **Restart the application** - Sometimes helps with initialization issues
        
        **Your current configuration:**
        - ChromaDB Path: `data/chroma_db`
        - Google API Key: `{'Set' if os.getenv('GOOGLE_API_KEY') else 'Not set'}`
        """)
    
    # Main content area
    if page == "📤 Document Upload":
        document_upload_page()
    elif page == "🔍 Semantic Search":
        semantic_search_page()
    elif page == "💬 Chatbot":
        chatbot_page()
    elif page == "📈 Analytics":
        analytics_page()

if __name__ == "__main__":
    main()
