
import streamlit as st
import os
import pickle
import numpy as np
from typing import List, Dict, Any
import google.generativeai as genai
from pathlib import Path
import tempfile
import shutil

# Document processing imports
import PyPDF2
from docx import Document
import pandas as pd
import openpyxl

# Load Gemini API key
def load_config():
    config = {}
    with open('application.properties', 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                config[key.strip()] = value.strip().strip('"')
    return config

config = load_config()
genai.configure(api_key=config['GEMINI_KEY'])

class DocumentProcessor:
    def __init__(self):
        self.supported_types = ['.pdf', '.docx', '.doc', '.xlsx', '.xls']
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
        return text
    
    def extract_text_from_excel(self, file_path: str) -> str:
        """Extract text from Excel file"""
        text = ""
        try:
            # Try reading with pandas first
            excel_file = pd.ExcelFile(file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                text += f"Sheet: {sheet_name}\n"
                text += df.to_string() + "\n\n"
        except Exception as e:
            st.error(f"Error reading Excel: {str(e)}")
        return text
    
    def process_document(self, file_path: str, file_extension: str) -> str:
        """Process document based on file type"""
        if file_extension.lower() == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension.lower() in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        elif file_extension.lower() in ['.xlsx', '.xls']:
            return self.extract_text_from_excel(file_path)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return ""

class GeminiEmbeddings:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Gemini"""
        try:
            # For Gemini 2.0 Flash, we'll use the embedding API
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            st.error(f"Error generating embedding: {str(e)}")
            return []
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query"""
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            st.error(f"Error generating query embedding: {str(e)}")
            return []

class RAGSystem:
    def __init__(self):
        self.embeddings_model = GeminiEmbeddings()
        self.doc_processor = DocumentProcessor()
        self.documents = []
        self.embeddings = []
        self.embeddings_file = "document_embeddings.pkl"
        self.documents_file = "documents.pkl"
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing embeddings and documents"""
        try:
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
            if os.path.exists(self.documents_file):
                with open(self.documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading existing data: {str(e)}")
            self.documents = []
            self.embeddings = []
    
    def save_data(self):
        """Save embeddings and documents to files"""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
        except Exception as e:
            st.error(f"Error saving data: {str(e)}")
    
    def add_document(self, file_name: str, text: str):
        """Add document and generate embedding"""
        if text.strip():
            # Split text into chunks if it's too long
            chunks = self.split_text(text, max_length=1000)
            
            for i, chunk in enumerate(chunks):
                embedding = self.embeddings_model.get_embedding(chunk)
                if embedding:
                    doc_info = {
                        'file_name': file_name,
                        'chunk_id': i,
                        'text': chunk,
                        'full_text': text
                    }
                    self.documents.append(doc_info)
                    self.embeddings.append(embedding)
            
            self.save_data()
            return len(chunks)
        return 0
    
    def split_text(self, text: str, max_length: int = 1000) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    chunks.append(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def search_similar_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if not self.embeddings:
            return []
        
        query_embedding = self.embeddings_model.get_query_embedding(query)
        if not query_embedding:
            return []
        
        # Calculate cosine similarity
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        results = []
        for similarity, idx in similarities[:top_k]:
            result = self.documents[idx].copy()
            result['similarity'] = similarity
            results.append(result)
        
        return results
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main():
    st.set_page_config(
        page_title="RAG Document Upload System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 RAG Document Upload & Search System")
    st.markdown("Upload documents (PDF, DOC, DOCX, XLS, XLSX) and search through them using Gemini 2.0 Flash embeddings")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    rag_system = st.session_state.rag_system
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'doc', 'xlsx', 'xls'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                progress_bar = st.progress(0)
                total_files = len(uploaded_files)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    st.write(f"Processing: {uploaded_file.name}")
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Extract text
                        file_extension = Path(uploaded_file.name).suffix
                        text = rag_system.doc_processor.process_document(tmp_file_path, file_extension)
                        
                        if text:
                            chunks_added = rag_system.add_document(uploaded_file.name, text)
                            st.success(f"Added {chunks_added} chunks from {uploaded_file.name}")
                        else:
                            st.error(f"Could not extract text from {uploaded_file.name}")
                    
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
                    
                    progress_bar.progress((idx + 1) / total_files)
                
                st.success("All documents processed!")
                st.rerun()
    
    # Main area for search and results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🔍 Search Documents")
        
        # Display current document count
        st.info(f"📊 Documents in database: {len(set(doc['file_name'] for doc in rag_system.documents))}")
        st.info(f"📝 Total chunks: {len(rag_system.documents)}")
        
        # Search interface
        search_query = st.text_area("Enter your search query:", height=100)
        top_k = st.slider("Number of results:", min_value=1, max_value=10, value=5)
        
        if st.button("Search") and search_query:
            with st.spinner("Searching..."):
                results = rag_system.search_similar_documents(search_query, top_k)
                st.session_state.search_results = results
    
    with col2:
        st.header("📋 Search Results")
        
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            for i, result in enumerate(st.session_state.search_results):
                with st.expander(f"Result {i+1}: {result['file_name']} (Similarity: {result['similarity']:.3f})"):
                    st.write("**Text snippet:**")
                    st.write(result['text'])
                    
                    if st.button(f"Show full document", key=f"show_full_{i}"):
                        st.text_area("Full document:", result['full_text'], height=300, key=f"full_text_{i}")
        
        elif hasattr(st.session_state, 'search_results'):
            st.info("No results found. Try a different query.")
        else:
            st.info("Enter a search query to see results here.")
    
    # Management section
    st.header("🗂️ Database Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear All Documents"):
            rag_system.documents = []
            rag_system.embeddings = []
            rag_system.save_data()
            st.success("All documents cleared!")
            st.rerun()
    
    with col2:
        if rag_system.documents:
            st.download_button(
                label="Download Document Database",
                data=pickle.dumps(rag_system.documents),
                file_name="document_database.pkl",
                mime="application/octet-stream"
            )

if __name__ == "__main__":
    main()
