import streamlit as st
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Import our offline RAG components
try:
    from ingestion.offline_extract import OfflineExtractor
    from ingestion.offline_embeddings import OfflineEmbeddingProcessor
    from retrieval.offline_search import OfflineSearch
    from retrieval.enhanced_offline_rag import EnhancedOfflineRAG
except ImportError as e:
    st.error(f"Failed to import offline RAG components: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Offline Multimodal RAG System",
    page_icon="🚀",
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
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .search-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .modality-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .score-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'source_files' not in st.session_state:
        st.session_state.source_files = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = {}

def setup_directories():
    """Create necessary directories"""
    directories = ["data/raw", "data/extracted", "data/faiss_db"]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_source_files():
    """Load all file names from data/raw into session state"""
    raw_dir = Path("data/raw")    
    files = [f.name for f in raw_dir.glob("*") if f.is_file()]
    st.session_state.source_files = files
    return files

def initialize_rag_system():
    """Initialize the offline RAG system"""
    if st.session_state.rag_system is not None:
        return st.session_state.rag_system
    
    try:
        rag_system = EnhancedOfflineRAG()
        
        # Check if system initialization was successful
        if not rag_system.search.processor:
            st.error("❌ Failed to initialize FAISS database. Please check your setup.")
            return None
        
        load_source_files()
        st.session_state.rag_system = rag_system
        return rag_system
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

def get_file_icon(file_type: str) -> str:
    """Get icon for file type"""
    icons = {
        'pdf': '📄',
        'docx': '📝',
        'image': '🖼️',
        'audio': '🎵',
        'text': '📝',
        'table': '📊',
        'unknown': '📁'
    }
    return icons.get(file_type, '📁')

def render_search_result(result: Dict[str, Any], index: int):
    """Render a single search result with enhanced UI"""
    with st.container():
        # Header with badges
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            file_icon = get_file_icon(result.get('type', 'unknown'))
            st.markdown(f"### {file_icon} Result {index + 1}")
        
        with col2:
            st.markdown(f'<span class="modality-badge">{result.get("type", "unknown").upper()}</span>', 
                       unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<span class="score-badge">Score: {result.get("score", 0):.3f}</span>', 
                       unsafe_allow_html=True)
        
        with col4:
            if st.button(f"📋 Copy", key=f"copy_{index}"):
                st.write("Content copied!")
        
        # Content preview
        content = result.get('text', '')
        if len(content) > 300:
            with st.expander("📖 Show full content", expanded=False):
                st.text(content)
            st.write(f"**Preview:** {content[:300]}...")
        else:
            st.write(f"**Content:** {content}")
        
        # Metadata in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**📁 Source:** {result.get('source_file', 'N/A')}")
        with col2:
            if result.get('page_number'):
                st.write(f"**📄 Page:** {result.get('page_number')}")
        with col3:
            if result.get('timestamp'):
                st.write(f"**⏱️ Time:** {result.get('timestamp'):.1f}s")
        with col4:
            if result.get('section_header'):
                st.write(f"**📑 Section:** {result.get('section_header')}")
        
        # Special content rendering
        if result.get('type') == 'image' and result.get('image_path'):
            render_image_result(result)
        elif result.get('type') == 'audio' and result.get('audio_path'):
            render_audio_result(result)
        
        st.markdown("---")

def render_image_result(result: Dict[str, Any]):
    """Render image result with preview"""
    image_path = result.get('image_path')
    if image_path and os.path.exists(image_path):
        try:
            with st.expander("🖼️ View Image", expanded=False):
                col1, col2 = st.columns([1, 2])
                with col1:
                    image = Image.open(image_path)
                    st.image(image, caption=f"Image from {result.get('source_file', 'Unknown')}", width=300)
                with col2:
                    if result.get('ocr_text'):
                        st.write("**OCR Text:**")
                        st.text(result['ocr_text'])
        except Exception as e:
            st.write(f"Error loading image: {e}")

def render_audio_result(result: Dict[str, Any]):
    """Render audio result with player"""
    audio_path = result.get('audio_path')
    if audio_path and os.path.exists(audio_path):
        try:
            with st.expander("🎵 Play Audio", expanded=False):
                st.audio(audio_path)
                if result.get('transcript'):
                    st.write("**Transcript:**")
                    st.text(result['transcript'])
        except Exception as e:
            st.write(f"Error loading audio: {e}")

def document_upload_page():
    """Enhanced document upload and ingestion page"""
    st.markdown('<h1 class="main-header">🚀 Offline Multimodal Document Processing</h1>', unsafe_allow_html=True)
    
    # File uploader with multiple formats
    st.subheader("📁 Upload Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose files (PDF, DOCX, Images, Audio)",
            type=['pdf', 'docx', 'doc', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 
                  'wav', 'mp3', 'm4a', 'flac', 'ogg', 'aac'],
            accept_multiple_files=True,
            help="Upload documents, images, and audio files for processing"
        )
    
    with col2:
        st.info("""
        **Supported Formats:**
        - 📄 PDF, DOCX
        - 🖼️ JPG, PNG, TIFF, etc.
        - 🎵 WAV, MP3, FLAC, etc.
        """)
    
    if uploaded_files:
        # Display file summary
        file_types = {}
        total_size = 0
        
        for file in uploaded_files:
            ext = Path(file.name).suffix.lower()
            file_type = 'unknown'
            
            if ext in ['.pdf']:
                file_type = 'pdf'
            elif ext in ['.docx', '.doc']:
                file_type = 'docx'
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                file_type = 'image'
            elif ext in ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']:
                file_type = 'audio'
            
            file_types[file_type] = file_types.get(file_type, 0) + 1
            total_size += file.size
        
        # File summary
        st.subheader("📊 Upload Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Files", len(uploaded_files))
        with col2:
            st.metric("PDF/DOCX", file_types.get('pdf', 0) + file_types.get('docx', 0))
        with col3:
            st.metric("Images", file_types.get('image', 0))
        with col4:
            st.metric("Audio", file_types.get('audio', 0))
        with col5:
            st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")
        
        # Processing button
        if st.button("🚀 Process All Files", type="primary", use_container_width=True):
            process_multimodal_files(uploaded_files)

def process_multimodal_files(uploaded_files):
    """Process uploaded multimodal files"""
    setup_directories()
    
    # Save uploaded files
    with st.spinner("Saving uploaded files..."):
        saved_files = []
        for file in uploaded_files:
            file_path = Path("data/raw") / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            saved_files.append(file_path)
        
        st.success(f"✅ Saved {len(saved_files)} files")
    
    # Process files
    with st.spinner("Processing multimodal content... This may take several minutes."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize extractor
            status_text.text("Initializing offline extractor...")
            progress_bar.progress(10)
            
            extractor = OfflineExtractor()
            
            # Extract content
            status_text.text("Extracting content from all files...")
            progress_bar.progress(30)
            
            all_chunks = {}
            total_chunks = 0
            
            for i, file_path in enumerate(saved_files):
                status_text.text(f"Processing {file_path.name}...")
                chunks = extractor.extract_file_content(file_path)
                if chunks:
                    all_chunks[str(file_path)] = chunks
                    total_chunks += len(chunks)
                
                progress_bar.progress(30 + (i + 1) * 40 // len(saved_files))
            
            # Initialize FAISS processor
            status_text.text("Initializing FAISS index...")
            progress_bar.progress(70)
            
            processor = OfflineEmbeddingProcessor()
            
            # Process and index chunks
            status_text.text("Generating embeddings and building index...")
            progress_bar.progress(80)
            
            total_indexed = 0
            chunk_stats = {'text': 0, 'image': 0, 'audio': 0, 'table': 0}
            
            for file_path, chunks in all_chunks.items():
                indexed = processor.add_chunks(chunks)
                total_indexed += indexed
                
                # Count by type
                for chunk in chunks:
                    chunk_type = chunk.chunk_type
                    if chunk_type in chunk_stats:
                        chunk_stats[chunk_type] += 1
            
            progress_bar.progress(100)
            status_text.text("Processing completed!")
            
            if total_indexed > 0:
                st.session_state.documents_processed = True
                st.session_state.processing_stats = {
                    'total_files': len(saved_files),
                    'total_chunks': total_chunks,
                    'total_indexed': total_indexed,
                    'chunk_stats': chunk_stats
                }
                
                # Show results
                st.markdown('<div class="success-message">', unsafe_allow_html=True)
                st.success("🎉 Offline multimodal processing completed successfully!")
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Files Processed", len(saved_files))
                with col2:
                    st.metric("Total Chunks", total_chunks)
                with col3:
                    st.metric("Indexed Chunks", total_indexed)
                with col4:
                    st.metric("Success Rate", f"{(total_indexed/total_chunks)*100:.1f}%")
                
                # Chunk distribution chart
                if chunk_stats:
                    st.subheader("📊 Content Distribution")
                    fig = px.pie(
                        values=list(chunk_stats.values()),
                        names=list(chunk_stats.keys()),
                        title="Chunks by Modality"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Processing failed: No chunks were indexed")
        
        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.exception(e)
        finally:
            progress_bar.empty()
            status_text.empty()

def multimodal_search_page():
    """Enhanced multimodal search page"""
    st.markdown('<h1 class="main-header">🔍 Offline Multimodal Search</h1>', unsafe_allow_html=True)

    if not st.session_state.documents_processed:
        st.warning("⚠️ Please upload and process documents first before searching.")
        return

    # Query complexity analysis
    if 'last_query' in st.session_state and st.session_state.last_query:
        with st.expander("🧠 Query Analysis", expanded=False):
            analysis = st.session_state.rag_system.analyze_query_complexity(st.session_state.last_query)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Question Type:** {', '.join(analysis['question_types'])}")
            with col2:
                st.write(f"**Complexity:** {analysis['complexity_score']:.2f}")
            with col3:
                st.write(f"**Recommended Results:** {analysis['recommended_top_k']}")
            
            if analysis['preferred_modalities']:
                st.write(f"**Preferred Modalities:** {', '.join(analysis['preferred_modalities'])}")

    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "🔍 Enter your search query:",
            placeholder="e.g., machine learning algorithms, neural network architecture, data visualization",
            help="Search across all modalities: text, images, audio, and tables"
        )
    
    with col2:
        search_type = st.selectbox(
            "Search Type",
            ["Standard", "Cross-Modal"],
            help="Standard: unified search, Cross-Modal: results grouped by type"
        )
    
    # Advanced filters
    with st.expander("🔧 Advanced Filters", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            modality_filter = st.selectbox(
                "Content Type",
                options=["All", "text", "image", "audio", "table"],
                help="Filter by content modality"
            )
        
        with col2:
            top_k = st.slider("Max Results", min_value=1, max_value=25, value=10)
        
        with col3:
            response_length = st.selectbox(
                "Response Length",
                options=["short", "medium", "long"],
                index=1,
                help="Length of AI-generated response"
            )
        
        with col4:
            source_file = st.selectbox(
                "Source File",
                options=["All Files"] + st.session_state.source_files,
                help="Filter by specific source file"
            )
        
        with col5:
            include_llm = st.checkbox("Include AI Response", value=True, 
                                    help="Generate AI response from search results")
    
    # Search buttons
    col1, col2 = st.columns(2)
    with col1:
        search_btn = st.button("🚀 Search", type="primary", use_container_width=True)
    with col2:
        if st.button("📊 Cross-Modal Analysis", use_container_width=True):
            search_type = "Cross-Modal"
            search_btn = True
    
    if search_btn and query and st.session_state.rag_system:
        st.session_state.last_query = query
        perform_search(query, search_type, modality_filter, top_k, source_file, include_llm, response_length)

def perform_search(query: str, search_type: str, modality_filter: str, top_k: int, source_file: str, include_llm: bool, response_length: str = "medium"):
    """Perform the actual search operation"""
    
    with st.spinner("🔍 Searching and generating AI response..."):
        try:
            # Prepare filters
            filters = {}
            if source_file != "All Files":
                filters['source_file'] = source_file
            
            # Perform search based on type
            if search_type == "Cross-Modal":
                result = st.session_state.rag_system.cross_modal_query(
                    query, 
                    top_k=top_k,
                    response_length=response_length
                )
                display_cross_modal_results(result)
            else:
                # Standard search
                modality = None if modality_filter == "All" else modality_filter
                result = st.session_state.rag_system.query(
                    question=query,
                    top_k=top_k,
                    modality_filter=modality,
                    include_llm_response=include_llm,
                    response_length=response_length
                )
                display_standard_results(result)
            
            # Add to search history
            st.session_state.search_history.append({
                "query": query,
                "type": search_type.lower(),
                "results_count": len(result.get('search_results', [])),
                "timestamp": time.time(),
                "modality_filter": modality_filter,
                "confidence": result.get('confidence', 0.0),
                "has_llm_response": bool(result.get('llm_response'))
            })
            
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.exception(e)

def display_standard_results(result: Dict[str, Any]):
    """Display standard search results"""
    
    # AI Response
    if result.get('llm_response'):
        st.subheader("🤖 AI-Generated Answer")
        
        # Show confidence and model info
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            confidence = result.get('confidence', 0.0)
            confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
            st.markdown(f"**Confidence:** <span style='color: {confidence_color}'>{confidence:.2f}</span>", 
                       unsafe_allow_html=True)
        with col2:
            model_info = result.get('metadata', {}).get('model_info', {})
            st.write(f"**Model:** {model_info.get('model_name', 'Unknown')}")
        with col3:
            citations_count = len(result.get('citations', []))
            st.write(f"**Citations:** {citations_count}")
        
        # Display the answer
        st.markdown(f'<div class="ai-message">{result["llm_response"]}</div>', 
                   unsafe_allow_html=True)
        
        # Show citations if available
        if result.get('citations'):
            with st.expander("📚 View Citations", expanded=False):
                for citation in result['citations']:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.write(f"**[{citation['id']}]**")
                    with col2:
                        st.write(f"**{citation['source_file']}**")
                        if citation.get('page_number'):
                            st.write(f"Page {citation['page_number']}")
                    with col3:
                        st.write(f"Type: {citation['type']}")
                        st.write(f"Score: {citation['score']:.3f}")
        
        st.markdown("---")
    
    # Search Results
    search_results = result.get('search_results', [])
    if search_results:
        metadata = result.get('metadata', {})
        total_found = metadata.get('total_results_found', len(search_results))
        st.subheader(f"🎯 Search Results ({len(search_results)} of {total_found} found)")
        
        for i, search_result in enumerate(search_results):
            render_search_result(search_result, i)
    else:
        st.info("No results found. Try a different query or adjust filters.")

def display_cross_modal_results(result: Dict[str, Any]):
    """Display cross-modal search results"""
    
    # AI Response
    if result.get('llm_response'):
        st.subheader("🤖 Cross-Modal AI Analysis")
        
        # Show enhanced metadata for cross-modal
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            confidence = result.get('confidence', 0.0)
            confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
            st.markdown(f"**Confidence:** <span style='color: {confidence_color}'>{confidence:.2f}</span>", 
                       unsafe_allow_html=True)
        with col2:
            citations_count = len(result.get('citations', []))
            st.write(f"**Citations:** {citations_count}")
        with col3:
            metadata = result.get('metadata', {})
            total_results = metadata.get('total_results', 0)
            st.write(f"**Total Found:** {total_results}")
        with col4:
            modality_counts = metadata.get('results_by_modality', {})
            active_modalities = len([k for k, v in modality_counts.items() if v > 0])
            st.write(f"**Modalities:** {active_modalities}/4")
        
        # Display the answer
        st.markdown(f'<div class="ai-message">{result["llm_response"]}</div>', 
                   unsafe_allow_html=True)
        
        # Enhanced citations for cross-modal
        if result.get('citations'):
            with st.expander("📚 Cross-Modal Citations", expanded=False):
                # Group citations by modality
                citations_by_modality = {}
                for citation in result['citations']:
                    modality = citation.get('type', 'text')
                    if modality not in citations_by_modality:
                        citations_by_modality[modality] = []
                    citations_by_modality[modality].append(citation)
                
                for modality, citations in citations_by_modality.items():
                    st.write(f"**{modality.upper()} Sources:**")
                    for citation in citations:
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            st.write(f"[{citation['id']}]")
                        with col2:
                            st.write(f"{citation['source_file']}")
                            if citation.get('page_number'):
                                st.write(f"Page {citation['page_number']}")
                        with col3:
                            st.write(f"Score: {citation['score']:.3f}")
                    st.write("")
        
        st.markdown("---")
    
    # Cross-modal results
    cross_results = result.get('cross_modal_results', {})
    
    if any(cross_results.values()):
        st.subheader("🔄 Cross-Modal Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Text", len(cross_results.get('text', [])))
        with col2:
            st.metric("🖼️ Images", len(cross_results.get('image', [])))
        with col3:
            st.metric("🎵 Audio", len(cross_results.get('audio', [])))
        with col4:
            st.metric("📊 Tables", len(cross_results.get('table', [])))
        
        # Results by modality
        for modality, results in cross_results.items():
            if results:
                with st.expander(f"{get_file_icon(modality)} {modality.upper()} Results ({len(results)})", 
                               expanded=True):
                    for i, res in enumerate(results):
                        render_search_result(res, i)
    else:
        st.info("No cross-modal results found.")

def chat_interface_page():
    """Enhanced chat interface with RAG"""
    st.markdown('<h1 class="main-header">💬 Offline AI Chat Assistant</h1>', unsafe_allow_html=True)
    
    if not st.session_state.documents_processed:
        st.warning("⚠️ Please upload and process documents first before using the chat.")
        return
    
    # Chat settings
    with st.sidebar:
        st.subheader("💬 Chat Settings")
        max_results = st.slider("Max Search Results", 1, 10, 5)
        response_length = st.selectbox("Response Length", ["short", "medium", "long"], index=1)
        include_sources = st.checkbox("Show Sources", value=True)
        chat_mode = st.selectbox("Chat Mode", ["Standard", "Cross-Modal"])
        show_confidence = st.checkbox("Show Confidence", value=True)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                st.markdown(f'<div class="chat-message user-message">👤 **You:** {message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message ai-message">🤖 **AI:** {message["content"]}</div>', 
                           unsafe_allow_html=True)
                
                # Add confidence indicator if available
                if show_confidence and 'confidence' in message:
                    confidence = message['confidence']
                    confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                    confidence_text = f" <span style='color: {confidence_color}; font-size: 0.8em;'>(Confidence: {confidence:.2f})</span>"
                    ai_header = f"🤖 **AI:**{confidence_text}"
                else:
                    ai_header = "🤖 **AI:**"
                
                st.markdown(f'<div class="chat-message ai-message">{ai_header} {message["content"]}</div>')
                # Show sources if available
                if include_sources and 'sources' in message:
                    with st.expander(f"📚 Sources ({len(message['sources'])})", expanded=False):
                        for j, source in enumerate(message['sources'], 1):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{j}. {source['source_file']}**")
                                if source.get('page_number'):
                                    st.write(f"   Page {source['page_number']} | Type: {source.get('type', 'text')}")
                                st.write(f"   {source['text'][:150]}...")
                            with col2:
                                st.write(f"Score: {source['score']:.3f}")
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your documents...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': time.time()
        })
        
        # Generate AI response
        with st.spinner("🤖 Analyzing documents and generating response..."):
            try:
                if chat_mode == "Cross-Modal":
                    result = st.session_state.rag_system.cross_modal_query(
                        user_input, 
                        top_k=max_results,
                        response_length=response_length
                    )
                    ai_response = result.get('llm_response', 'Sorry, I could not generate a response.')
                    sources = result.get('combined_results', [])
                    confidence = result.get('confidence', 0.0)
                else:
                    result = st.session_state.rag_system.query(
                        user_input, 
                        top_k=max_results,
                        response_length=response_length
                    )
                    ai_response = result.get('llm_response', 'Sorry, I could not generate a response.')
                    sources = result.get('search_results', [])
                    confidence = result.get('confidence', 0.0)
                
                # Add AI response to history
                ai_message = {
                    'role': 'assistant',
                    'content': ai_response,
                    'timestamp': time.time(),
                    'confidence': confidence
                }
                
                if include_sources and sources:
                    ai_message['sources'] = sources[:3]  # Top 3 sources
                
                st.session_state.chat_history.append(ai_message)
                
                # Rerun to show new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to generate response: {e}")
                # Add error message to chat
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': f"I encountered an error: {str(e)}. Please try rephrasing your question.",
                    'timestamp': time.time(),
                    'confidence': 0.0
                })

def analytics_dashboard():
    """Enhanced analytics and statistics dashboard"""
    st.markdown('<h1 class="main-header">📈 Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    if not st.session_state.documents_processed:
        st.warning("⚠️ No data to analyze. Please upload and process documents first.")
        return
    
    # System Statistics
    if st.session_state.rag_system:
        stats = st.session_state.rag_system.get_system_stats()
        
        st.subheader("🖥️ System Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Status", stats.get('system_status', 'unknown').upper())
        with col2:
            st.metric("LLM Model", stats.get('llm_model', 'Unknown'))
        with col3:
            total_vectors = stats.get('search_index', {}).get('total_vectors', 0)
            st.metric("Total Vectors", total_vectors)
        with col4:
            index_size = stats.get('search_index', {}).get('index_size_mb', 0)
            st.metric("Index Size", f"{index_size:.1f} MB")
    
    # Processing Statistics
    if st.session_state.processing_stats:
        st.subheader("📊 Processing Statistics")
        
        stats = st.session_state.processing_stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Files Processed", stats.get('total_files', 0))
        with col2:
            st.metric("Total Chunks", stats.get('total_chunks', 0))
        with col3:
            st.metric("Indexed Chunks", stats.get('total_indexed', 0))
        with col4:
            success_rate = (stats.get('total_indexed', 0) / max(stats.get('total_chunks', 1), 1)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Content distribution
        chunk_stats = stats.get('chunk_stats', {})
        if chunk_stats:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=list(chunk_stats.values()),
                    names=list(chunk_stats.keys()),
                    title="Content Distribution by Modality"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=list(chunk_stats.keys()),
                    y=list(chunk_stats.values()),
                    title="Chunks by Content Type"
                )
                fig.update_layout(xaxis_title="Content Type", yaxis_title="Number of Chunks")
                st.plotly_chart(fig, use_container_width=True)
    
    # Search Analytics
    if st.session_state.search_history:
        st.subheader("🔍 Search Analytics")
        
        history_df = pd.DataFrame(st.session_state.search_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], unit='s')
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Searches", len(history_df))
        with col2:
            avg_results = history_df['results_count'].mean() if len(history_df) > 0 else 0
            st.metric("Avg Results per Search", f"{avg_results:.1f}")
        with col3:
            recent_searches = len(history_df[history_df['timestamp'] > pd.Timestamp.now() - pd.Timedelta(hours=24)])
            st.metric("Searches (24h)", recent_searches)
        
        # Search history table
        st.subheader("📋 Recent Searches")
        display_df = history_df[['query', 'type', 'results_count', 'modality_filter', 'timestamp']].head(10)
        st.dataframe(display_df, use_container_width=True)

def main():
    """Main Streamlit application"""
    initialize_session_state()
    setup_directories()
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🚀 Offline Multimodal RAG")

        st.markdown("---")
        
        page = st.selectbox(
            "🧭 Navigation:",
            [
                "📤 Document Upload",
                "🔍 Multimodal Search", 
                "💬 AI Chat Assistant",
                "📈 Analytics Dashboard"
            ]
        )
        
        st.markdown("---")
        st.subheader("🖥️ System Status")
        
        # System status indicators
        if rag_system and rag_system.search.processor:
            st.success("✅ RAG System Ready")
        else:
            st.error("❌ RAG System Error")
        
        if st.session_state.documents_processed:
            st.success("✅ Documents Processed")
            
            # Show processing stats if available
            if st.session_state.processing_stats:
                stats = st.session_state.processing_stats
                st.info(f"📊 {stats.get('total_indexed', 0)} chunks indexed")
        else:
            st.info("ℹ️ No Documents")
        
        # Quick stats
        if st.session_state.search_history:
            st.info(f"🔍 {len(st.session_state.search_history)} searches performed")
        
        st.markdown("---")
        st.markdown("""
        ### 🎯 Features
        - **📄 PDF & DOCX** processing
        - **🖼️ Image OCR** with CLIP embeddings  
        - **🎵 Audio transcription** with Whisper
        - **🔍 Cross-modal search** with FAISS
        - **🤖 Enhanced offline LLM** with quantization
        - **📊 Real-time analytics**
        - **🔒 Fully offline** operation
        - **📚 Citation support** with confidence scores
        - **🧠 Query complexity analysis**
        """)
        
        
    
    # Main content area
    if page == "📤 Document Upload":
        document_upload_page()
    elif page == "🔍 Multimodal Search":
        multimodal_search_page()
    elif page == "💬 AI Chat Assistant":
        chat_interface_page()
    elif page == "📈 Analytics Dashboard":
        analytics_dashboard()

if __name__ == "__main__":
    main()