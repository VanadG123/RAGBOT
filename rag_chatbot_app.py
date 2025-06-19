import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.schema import Document
from typing import List
import uuid
import tempfile
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SentenceTransformerEmbeddings(Embeddings):
    """Custom embedding class for Sentence Transformers integration with LangChain"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode([text])
        return embedding[0].tolist()

class RAGChatbot:
    """Complete RAG Chatbot with ChromaDB, Sentence Transformers, and OpenAI"""

    def __init__(self):
        # Initialize components
        self.embedding_model = SentenceTransformerEmbeddings()
        self.chroma_client = None
        self.collection = None
        self.vectorstore = None
        self.conversation_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create persistent ChromaDB client
        self.persist_directory = "./chroma_db"
        self.setup_chromadb()

    def setup_chromadb(self):
        """Initialize ChromaDB with persistent storage"""
        try:
            # Create ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(allow_reset=True)
            )

            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection("company_knowledge")
                st.success("Loaded existing knowledge base!")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="company_knowledge",
                    metadata={"description": "Company knowledge base for RAG"}
                )
                st.info("Created new knowledge base.")

        except Exception as e:
            st.error(f"Error setting up ChromaDB: {str(e)}")

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF using PyPDF2"""
        try:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""

    def extract_text_from_csv(self, csv_file) -> str:
        """Extract text from CSV using Pandas"""
        try:
            df = pd.read_csv(csv_file)
            # Convert DataFrame to text representation
            text = ""
            for _, row in df.iterrows():
                # Create a readable text representation of each row
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                text += row_text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
        """Split text into chunks using RecursiveCharacterTextSplitter"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def add_documents_to_vectorstore(self, documents: List[str], metadata: List[dict] = None):
        """Add documents to ChromaDB with embeddings"""
        try:
            if not documents:
                return

            # Generate embeddings
            embeddings = self.embedding_model.embed_documents(documents)

            # Generate IDs for documents
            ids = [str(uuid.uuid4()) for _ in documents]

            # Prepare metadata
            if metadata is None:
                metadata = [{"source": "uploaded_document"} for _ in documents]

            # Add to ChromaDB collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadata
            )

            st.success(f"Added {len(documents)} document chunks to knowledge base!")

        except Exception as e:
            st.error(f"Error adding documents to vectorstore: {str(e)}")

    def setup_langchain_vectorstore(self):
        """Setup LangChain-compatible vectorstore from ChromaDB"""
        try:
            # Create LangChain Chroma vectorstore
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name="company_knowledge",
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
            return True
        except Exception as e:
            st.error(f"Error setting up LangChain vectorstore: {str(e)}")
            return False

    def create_conversation_chain(self, openai_api_key: str):
        """Create conversational RAG chain with memory"""
        try:
            if not self.setup_langchain_vectorstore():
                return False

            # Initialize OpenAI LLM
            llm = OpenAI(
                temperature=0.7,
                openai_api_key=openai_api_key,
                model_name="gpt-3.5-turbo-instruct"
            )

            # Create conversational retrieval chain
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )

            return True

        except Exception as e:
            st.error(f"Error creating conversation chain: {str(e)}")
            return False

    def process_uploaded_files(self, uploaded_files):
        """Process uploaded PDF and CSV files"""
        all_chunks = []
        all_metadata = []

        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                if file_type == "application/pdf":
                    # Process PDF
                    with open(tmp_file_path, 'rb') as pdf_file:
                        text = self.extract_text_from_pdf(pdf_file)
                elif file_type == "text/csv":
                    # Process CSV
                    text = self.extract_text_from_csv(tmp_file_path)
                else:
                    st.warning(f"Unsupported file type: {file_type}")
                    continue

                if text.strip():
                    # Chunk the text
                    chunks = self.chunk_text(text)

                    # Create metadata for each chunk
                    chunk_metadata = [
                        {
                            "source": uploaded_file.name,
                            "file_type": file_type,
                            "chunk_index": i
                        }
                        for i in range(len(chunks))
                    ]

                    all_chunks.extend(chunks)
                    all_metadata.extend(chunk_metadata)

                    st.success(f"Processed {uploaded_file.name}: {len(chunks)} chunks")

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)

        if all_chunks:
            # Add to vectorstore
            self.add_documents_to_vectorstore(all_chunks, all_metadata)
            return True

        return False

    def add_company_data(self, company_data_text: str):
        """Add pre-existing company data to the knowledge base"""
        if company_data_text.strip():
            chunks = self.chunk_text(company_data_text)
            metadata = [{"source": "company_data", "type": "pre_loaded"} for _ in chunks]
            self.add_documents_to_vectorstore(chunks, metadata)
            return True
        return False

    def get_response(self, question: str) -> tuple:
        """Get response from the conversational RAG chain"""
        try:
            if not self.conversation_chain:
                return "Please setup the conversation chain first by providing your OpenAI API key.", []

            # Get response from the chain
            result = self.conversation_chain({"question": question})

            # Extract answer and source documents
            answer = result.get("answer", "I couldn't generate a response.")
            source_docs = result.get("source_documents", [])

            return answer, source_docs

        except Exception as e:
            return f"Error generating response: {str(e)}", []

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="RAG Chatbot with Company Knowledge",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 RAG-Based Chatbot with Company Knowledge")
    st.markdown("### Upload PDFs/CSVs and chat with your documents using AI!")

    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "conversation_ready" not in st.session_state:
        st.session_state.conversation_ready = False

    # Sidebar for configuration and file upload
    with st.sidebar:
        st.header("⚙️ Configuration")

        # OpenAI API Key input
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable the chatbot"
        )

        if openai_api_key and not st.session_state.conversation_ready:
            if st.session_state.chatbot.create_conversation_chain(openai_api_key):
                st.session_state.conversation_ready = True
                st.success("✅ Chatbot ready!")
            else:
                st.error("❌ Failed to setup chatbot")

        st.header("📁 Upload Documents")

        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF or CSV files",
            type=["pdf", "csv"],
            accept_multiple_files=True,
            help="Upload PDF documents or CSV files to add to the knowledge base"
        )

        if uploaded_files and st.button("Process Files"):
            with st.spinner("Processing uploaded files..."):
                if st.session_state.chatbot.process_uploaded_files(uploaded_files):
                    st.success("Files processed successfully!")
                    # Reset conversation chain to include new documents
                    if openai_api_key:
                        st.session_state.chatbot.create_conversation_chain(openai_api_key)

        st.header("🏢 Company Data")

        # Company data input
        company_data = st.text_area(
            "Pre-load Company Knowledge",
            height=200,
            help="Enter any company-specific information, policies, or knowledge that should be available to the chatbot",
            placeholder="Enter company policies, product information, or any other relevant knowledge..."
        )

        if company_data and st.button("Add Company Data"):
            with st.spinner("Adding company data..."):
                if st.session_state.chatbot.add_company_data(company_data):
                    st.success("Company data added!")
                    # Reset conversation chain to include new data
                    if openai_api_key:
                        st.session_state.chatbot.create_conversation_chain(openai_api_key)

        # Knowledge base stats
        st.header("📊 Knowledge Base Stats")
        try:
            collection = st.session_state.chatbot.collection
            if collection:
                count = collection.count()
                st.metric("Total Documents", count)
        except:
            st.metric("Total Documents", "N/A")

    # Main chat interface
    st.header("💬 Chat Interface")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**Source {i+1}:** {source.metadata.get('source', 'Unknown')}")
                        st.write(f"*Content:* {source.page_content[:200]}...")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.conversation_ready:
            st.error("Please provide your OpenAI API key to start chatting!")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, sources = st.session_state.chatbot.get_response(prompt)
                st.markdown(response)

                # Show sources if available
                if sources:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(sources):
                            st.write(f"**Source {i+1}:** {source.metadata.get('source', 'Unknown')}")
                            st.write(f"*Content:* {source.page_content[:200]}...")

            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources
            })

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if st.session_state.chatbot.memory:
            st.session_state.chatbot.memory.clear()
        st.rerun()

if __name__ == "__main__":
    main()