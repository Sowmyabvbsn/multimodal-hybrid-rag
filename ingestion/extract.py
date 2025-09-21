import base64
from io import BytesIO
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image
import pandas as pd

try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import (
        Table, Title, Header
    )
    UNSTRUCTURED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unstructured library not available: {e}")
    UNSTRUCTURED_AVAILABLE = False
    # Fallback imports
    import fitz  # PyMuPDF

@dataclass
class ExtractedChunk:
    """Container for extracted PDF chunk with metadata"""
    content: str
    chunk_type: str  # 'text', 'table', 'image'
    page_number: int
    section_header: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    table_data: Optional[str] = None # HTML representation of table

class PDFExtractor:
    """Extract multimodal content from PDF documents"""
    
    def __init__(self, raw_dir: str = "data/raw", extracted_dir: str = "data/extracted"):
        self.raw_dir = Path(raw_dir)
        self.extracted_dir = Path(extracted_dir)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
            
    def get_pdf_hash(self, pdf_path: Path) -> str:
        """Generate SHA-256 hash of PDF file"""
        hash_sha256 = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()[:16]  # Use first 16 chars
    
    def extract_pdf_content(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract all content from PDF using Unstructured"""
        
        if not UNSTRUCTURED_AVAILABLE:
            return self._extract_with_pymupdf(pdf_path)
        
        # Create output directory for this PDF
        pdf_hash = self.get_pdf_hash(pdf_path)
        output_dir = self.extracted_dir / f"{pdf_path.stem}_{pdf_hash}"
        output_dir.mkdir(exist_ok=True)
        image_path_dir = output_dir / "images"
        image_path_dir.mkdir(exist_ok=True)
        
        # Extract using Unstructured
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_output_dir=str(image_path_dir),
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
        )
        
        chunks = []

        texts = []
        tables = []
        images = []
        image_counter = 0
        current_section_header = None

        for element in elements:
            # Process different element types
            if hasattr(element.metadata, "orig_elements") and element.metadata.orig_elements:
                for elem in element.metadata.orig_elements:

                    if isinstance(elem, Table):
                        # Table content
                        table_data = elem.metadata.text_as_html
                        content = elem.text
                        page_number=elem.metadata.page_number
                        section_header = current_section_header if current_section_header else self._get_section_header(element.metadata.orig_elements, elem)

                        chunk = ExtractedChunk(
                            content=content,
                            chunk_type="table",
                            page_number=page_number,
                            section_header=section_header,
                            table_data=table_data,
                            metadata={
                                "document_id": pdf_hash,
                                "source_file": pdf_path.name,
                            }
                        )
                        tables.append(chunk)
                        chunks.append(chunk)

                    # elif isinstance(elem, Image):
                    elif 'Image' in str(type(elem)):
                        # Image content
                        image_base64 = elem.metadata.image_base64
                        image_mime_type = elem.metadata.image_mime_type
                        page_number=elem.metadata.page_number

                        # Save base64 image to file
                        saved_image_path = self._save_base64_image(
                            image_base64, image_path_dir, image_counter, page_number, image_mime_type
                        )

                        section_header = current_section_header if current_section_header else self._get_section_header(element.metadata.orig_elements, elem)

                        content = elem.text.strip() if hasattr(elem, 'text') else f"Image from page {page_number}"

                        chunk = ExtractedChunk(
                            content=content,
                            chunk_type="image",
                            page_number=page_number,
                            section_header=section_header,
                            image_path=saved_image_path,
                            image_base64=image_base64,
                            metadata={
                                # "image_format": image_path.suffix,
                                "document_id": pdf_hash,
                                "source_file": pdf_path.name,
                            }
                        )
                        images.append(chunk)
                        chunks.append(chunk)
                        image_counter += 1

                    else:
                        # Text content
                        if isinstance(elem, (Title, Header)):
                            current_section_header = elem.text.strip()
                            section_header = current_section_header if current_section_header else self._get_section_header(element.metadata.orig_elements, elem)

                        content = elem.text
                        page_number=elem.metadata.page_number

                        chunk = ExtractedChunk(
                            content=content,
                            chunk_type="text",
                            page_number=page_number,
                            section_header=section_header,
                            metadata={
                                "document_id": pdf_hash,
                                "source_file": pdf_path.name,
                            }
                        )
                        texts.append(chunk)
                        chunks.append(chunk)

        # Save extracted chunks metadata
        self._save_chunks_metadata(chunks, output_dir)
        metadata = {
            "document_id": pdf_hash,
            "source_file": pdf_path.name,
        }
        # Save chunk data grouped by page and type
        chunks_data = self._save_chunk_data(chunks, metadata, output_dir)
        
        return chunks_data

    def _get_section_header(self, elements, current_element) -> Optional[str]:
        """Try to identify section header for current element"""
        current_idx = None
        for i, elem in enumerate(elements):
            if elem == current_element:
                current_idx = i
                break
        
        if current_idx is not None:
            # Look for header/title elements before current element on same page
            current_page = getattr(current_element.metadata, 'page_number', 1)
            for i in range(current_idx - 1, -1, -1):
                elem = elements[i]
                elem_page = getattr(elem.metadata, 'page_number', 1)
                
                if elem_page != current_page:
                    break
                
                if isinstance(elem, (Title, Header)) and len(str(elem)) < 200:
                    return str(elem.text).strip()
        
        return None
    
    def _save_base64_image(self, image_base64: str, image_path_dir: Path, image_counter: int, page_number: int, mime_type: str = None) -> Optional[str]:
        """Decode base64 image and save to file"""
        if not image_base64:
            return None
        
        try:
            # Remove data URL prefix if present
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(image_base64)
            
            # Determine file extension from mime type
            if mime_type:
                format_ext = mime_type.split('/')[-1].lower()
                # Handle common variations
                if format_ext == 'jpeg':
                    format_ext = 'jpg'
            else:
                # Fallback to PIL detection
                img = Image.open(BytesIO(image_data))
                format_ext = img.format.lower() if img.format else 'png'
            
            # Create filename
            filename = f"page_{page_number}_image_{image_counter}.{format_ext}"
            image_path = image_path_dir / filename
            
            # Save image
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            return str(image_path)
            
        except Exception as e:
            return None

    def _save_chunk_data(self, chunks: List[ExtractedChunk], metadata: Dict, output_dir: Path):
        """Save chunks data grouped by page and type"""
        # Group chunks by page number
        page_chunks = {}
        for chunk in chunks:
            page_num = chunk.page_number
            if page_num not in page_chunks:
                metadata_with_page = metadata.copy()
                metadata_with_page['page_number'] = page_num
                page_chunks[page_num] = {"text": "", "image": [], "table": [], "metadata": metadata_with_page}
            

            if chunk.chunk_type == "text":
                # Concatenate all text content for the page
                page_chunks[page_num]["text"] += chunk.content + " "
            
            elif chunk.chunk_type == "table":
                table_data = {
                    "content": chunk.content,
                    "section_header": chunk.section_header,
                    "table_data": chunk.table_data
                }
                page_chunks[page_num]["table"].append(table_data)

            elif chunk.chunk_type == "image":

                image_data = {
                    "content": chunk.content,
                    "section_header": chunk.section_header,
                    "image_path": chunk.image_path,
                    "image_base64": chunk.image_base64
                }
                page_chunks[page_num]["image"].append(image_data)
        
        
        # Convert to list format ordered by page number
        chunks_data = []
        for page_num in sorted(page_chunks.keys()):
            page_data = page_chunks[page_num]
            # Clean up text (remove extra spaces)
            page_data["text"] = page_data["text"].strip()
            chunks_data.append(page_data)
        
        # Save to file
        chunk_data_path = output_dir / "chunk_data.json"
        with open(chunk_data_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        return chunks_data

    def _extract_with_pymupdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Fallback extraction using PyMuPDF when Unstructured is not available"""
        
        # Create output directory for this PDF
        pdf_hash = self.get_pdf_hash(pdf_path)
        output_dir = self.extracted_dir / f"{pdf_path.stem}_{pdf_hash}"
        output_dir.mkdir(exist_ok=True)
        image_path_dir = output_dir / "images"
        image_path_dir.mkdir(exist_ok=True)
        
        chunks = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text
            text = page.get_text()
            if text.strip():
                chunk = {
                    "content": text,
                    "chunk_type": "text",
                    "page_number": page_num + 1,
                    "section_header": None,
                    "metadata": {
                        "document_id": pdf_hash,
                        "source_file": pdf_path.name,
                    }
                }
                chunks.append(chunk)
            
            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_filename = f"page_{page_num + 1}_image_{img_index}.png"
                        img_path = image_path_dir / img_filename
                        
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                        
                        chunk = {
                            "content": f"Image from page {page_num + 1}",
                            "chunk_type": "image",
                            "page_number": page_num + 1,
                            "section_header": None,
                            "image_path": str(img_path),
                            "metadata": {
                                "document_id": pdf_hash,
                                "source_file": pdf_path.name,
                            }
                        }
                        chunks.append(chunk)
                    
                    pix = None
                except Exception as e:
                    print(f"Error extracting image: {e}")
                    continue
        
        doc.close()
        
        # Save extracted chunks metadata
        self._save_chunks_metadata_pymupdf(chunks, output_dir)
        
        # Save chunk data grouped by page and type
        metadata = {
            "document_id": pdf_hash,
            "source_file": pdf_path.name,
        }
        chunks_data = self._save_chunk_data_pymupdf(chunks, metadata, output_dir)
        
        return chunks_data
    
    def _save_chunks_metadata_pymupdf(self, chunks: List[Dict], output_dir: Path):
        """Save chunks metadata as JSON for PyMuPDF extraction"""
        chunks_data = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": i,
                "content": chunk["content"][:500] + "..." if len(chunk["content"]) > 500 else chunk["content"],
                "chunk_type": chunk["chunk_type"],
                "page_number": chunk["page_number"],
                "section_header": chunk.get("section_header"),
                "metadata": chunk["metadata"],
                "image_path": chunk.get("image_path"),
            }
            chunks_data.append(chunk_data)
        
        metadata_path = output_dir / "chunks_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    def _save_chunk_data_pymupdf(self, chunks: List[Dict], metadata: Dict, output_dir: Path):
        """Save chunks data grouped by page and type for PyMuPDF extraction"""
        # Group chunks by page number
        page_chunks = {}
        for chunk in chunks:
            page_num = chunk["page_number"]
            if page_num not in page_chunks:
                metadata_with_page = metadata.copy()
                metadata_with_page['page_number'] = page_num
                page_chunks[page_num] = {"text": "", "image": [], "table": [], "metadata": metadata_with_page}
            
            if chunk["chunk_type"] == "text":
                page_chunks[page_num]["text"] += chunk["content"] + " "
            elif chunk["chunk_type"] == "image":
                image_data = {
                    "content": chunk["content"],
                    "section_header": chunk.get("section_header"),
                    "image_path": chunk.get("image_path"),
                }
                page_chunks[page_num]["image"].append(image_data)
        
        # Convert to list format ordered by page number
        chunks_data = []
        for page_num in sorted(page_chunks.keys()):
            page_data = page_chunks[page_num]
            page_data["text"] = page_data["text"].strip()
            chunks_data.append(page_data)
        
        # Save to file
        chunk_data_path = output_dir / "chunk_data.json"
        with open(chunk_data_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        return chunks_data

    def _save_chunks_metadata(self, chunks: List[ExtractedChunk], output_dir: Path):
        """Save chunks metadata as JSON"""
        chunks_data = []
        for i, chunk in enumerate(chunks):
            image_path = chunk.image_path
            if image_path and output_dir in Path(image_path).parents:
                image_path = str(Path(image_path).relative_to(output_dir))
        
            chunk_data = {
                "chunk_id": i,
                "content": chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
                "chunk_type": chunk.chunk_type,
                "page_number": chunk.page_number,
                "section_header": chunk.section_header,
                "metadata": chunk.metadata,
                "image_path": chunk.image_path,
                "image_base64": chunk.image_base64[:30] + "..." if chunk.image_base64 and len(chunk.image_base64) > 30 else chunk.image_base64,
                "table_data": chunk.table_data
            }
            chunks_data.append(chunk_data)
        
        metadata_path = output_dir / "chunks_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
    
    def extract_all_pdfs(self) -> Dict[str, List[ExtractedChunk]]:
        """Extract content from all PDFs in raw directory"""
        pdf_files = list(self.raw_dir.glob("*.pdf"))
        
        if not pdf_files:
            return {}
        
        all_chunks = {}
        for pdf_path in pdf_files:
            try:
                chunks = self.extract_pdf_content(pdf_path)
                all_chunks[str(pdf_path)] = chunks
            except Exception as e:
                return {}
        
        return all_chunks

if __name__ == "__main__":
    # Example usage
    extractor = PDFExtractor()
    # all_chunks = extractor.extract_all_pdfs()
    
    # for pdf_path, chunks in all_chunks.items():
    #     print(f"\n{pdf_path}: {len(chunks)} chunks")
    #     for chunk in chunks[:3]:  # Show first 3 chunks
    #         print(f"  - {chunk.chunk_type}: {chunk.content[:100]}...")
