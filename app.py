
import streamlit as st
import os
import tempfile
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from docx import Document
from pptx import Presentation
import PyPDF2
import fitz  # PyMuPDF

class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_file = "document_embeddings.pkl"
        self.documents_data = []
        self.load_existing_embeddings()
    
    def load_existing_embeddings(self):
        """Load existing embeddings if they exist"""
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                self.documents_data = pickle.load(f)
    
    def save_embeddings(self):
        """Save embeddings to file"""
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.documents_data, f)
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        text = ""
        try:
            # Try with PyMuPDF first
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except:
            # Fallback to PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        return text
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX file"""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def extract_text_from_pptx(self, file_path):
        """Extract text from PPTX file"""
        prs = Presentation(file_path)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text
    
    def process_document(self, uploaded_file):
        """Process uploaded document and create embeddings"""
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Extract text based on file type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                text = self.extract_text_from_pdf(tmp_file_path)
            elif file_extension in ['doc', 'docx']:
                text = self.extract_text_from_docx(tmp_file_path)
            elif file_extension in ['ppt', 'pptx']:
                text = self.extract_text_from_pptx(tmp_file_path)
            else:
                return False, "Unsupported file type"
            
            if not text.strip():
                return False, "No text could be extracted from the document"
            
            # Create chunks (simple sentence-based chunking)
            sentences = text.split('.')
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) < 500:  # Keep chunks under 500 chars
                    current_chunk += sentence + "."
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + "."
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Remove empty chunks
            chunks = [chunk for chunk in chunks if chunk.strip()]
            
            if not chunks:
                return False, "No meaningful content could be extracted"
            
            # Create embeddings
            embeddings = self.model.encode(chunks)
            
            # Store document data
            doc_data = {
                'filename': uploaded_file.name,
                'chunks': chunks,
                'embeddings': embeddings,
                'file_type': file_extension
            }
            
            self.documents_data.append(doc_data)
            self.save_embeddings()
            
            return True, f"Successfully processed {len(chunks)} chunks from {uploaded_file.name}"
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    def search_documents(self, query, top_k=5):
        """Search for relevant document chunks"""
        if not self.documents_data:
            return []
        
        query_embedding = self.model.encode([query])
        
        results = []
        for doc_data in self.documents_data:
            # Calculate similarity scores
            similarities = np.dot(query_embedding, doc_data['embeddings'].T).flatten()
            
            # Get top chunks for this document
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Threshold for relevance
                    results.append({
                        'filename': doc_data['filename'],
                        'chunk': doc_data['chunks'][idx],
                        'score': similarities[idx]
                    })
        
        # Sort all results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

def main():
    st.set_page_config(page_title="RAG Document Upload System", page_icon="📚", layout="wide")
    
    st.title("📚 RAG Document Upload & Search System")
    st.markdown("Upload documents (PDF, DOC, PPT) to create embeddings for Retrieval-Augmented Generation")
    
    # Initialize processor
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    
    processor = st.session_state.processor
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'doc', 'docx', 'ppt', 'pptx'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOC, DOCX, PPT, PPTX"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                progress_bar = st.progress(0)
                status_container = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_container.text(f"Processing {uploaded_file.name}...")
                    success, message = processor.process_document(uploaded_file)
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(f"Error processing {uploaded_file.name}: {message}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_container.text("Processing complete!")
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🔍 Search Documents")
        query = st.text_input("Enter your search query:", placeholder="What would you like to find?")
        
        if query and st.button("Search", type="primary"):
            results = processor.search_documents(query)
            st.session_state.search_results = results
        
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            st.subheader("Search Results")
            for i, result in enumerate(st.session_state.search_results):
                with st.expander(f"Result {i+1} - {result['filename']} (Score: {result['score']:.3f})"):
                    st.write(result['chunk'])
    
    with col2:
        st.header("📋 Document Library")
        
        if processor.documents_data:
            st.write(f"**Total Documents:** {len(processor.documents_data)}")
            
            for doc_data in processor.documents_data:
                with st.expander(f"📄 {doc_data['filename']} ({doc_data['file_type'].upper()})"):
                    st.write(f"**Chunks:** {len(doc_data['chunks'])}")
                    st.write(f"**File Type:** {doc_data['file_type'].upper()}")
                    
                    if st.button(f"Show chunks for {doc_data['filename']}", key=f"show_{doc_data['filename']}"):
                        st.write("**Document Chunks:**")
                        for i, chunk in enumerate(doc_data['chunks'][:5]):  # Show first 5 chunks
                            st.write(f"**Chunk {i+1}:** {chunk[:200]}...")
        else:
            st.info("No documents uploaded yet. Use the sidebar to upload documents.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** Embeddings are stored locally in `document_embeddings.pkl`")

if __name__ == "__main__":
    main()
