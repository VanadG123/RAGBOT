Running the Application
Local Development
bash
streamlit run rag_chatbot_app.py
The application will open in your browser at http://localhost:8501

Production Deployment
Streamlit Community Cloud
Push your code to a GitHub repository

Go to Streamlit Community Cloud

Click "New app" and connect your repository

Add your OpenAI API key in the Secrets section:

text
OPENAI_API_KEY = "your_api_key_here"
Deploy!

Other Deployment Options
Heroku: Use the provided requirements.txt

Docker: Create a Dockerfile based on the Python image

AWS/GCP/Azure: Deploy as a containerized application

📖 Usage Guide
1. Initial Setup
Configure API Key: Enter your OpenAI API key in the sidebar

Wait for Confirmation: Look for the "✅ Chatbot ready!" message

2. Building Your Knowledge Base
Option A: Upload Documents
Use the "Upload Documents" section in the sidebar

Select PDF files or CSV files (multiple files supported)

Click "Process Files" to add them to the knowledge base

Option B: Add Company Data
Use the "Company Data" section in the sidebar

Enter company policies, product information, or any relevant knowledge

Click "Add Company Data" to include it in the knowledge base

3. Chatting with Your Documents
Use the chat interface in the main area

Ask questions about your uploaded documents or company data

View sources used for each response in the expandable "Sources" section

Enjoy conversational context - the bot remembers your conversation history!

4. Managing Your Knowledge Base
View Stats: Check the "Knowledge Base Stats" in the sidebar to see how many documents are indexed

Clear Chat: Use the "Clear Chat History" button to start fresh

Add More Data: You can continuously add more documents and company data

🏗️ Architecture Overview
Data Flow
text
PDF/CSV Upload → Text Extraction → Chunking (512 tokens, 50 overlap) → 
Sentence-Transformers Embedding → ChromaDB Storage → 
LangChain Retrieval → OpenAI Generation → Streamlit Display
Key Components
SentenceTransformerEmbeddings: Custom LangChain-compatible embedding class

RAGChatbot: Main class handling all RAG operations

ChromaDB Integration: Persistent vector storage with metadata

Conversational Chain: LangChain's ConversationalRetrievalChain with memory

Streamlit Interface: User-friendly web interface

Technical Specifications
Embedding Model: all-MiniLM-L6-v2 (384 dimensions, 512 max tokens)

Chunk Size: 512 tokens with 50 token overlap

Vector Database: ChromaDB with persistent storage

LLM: OpenAI GPT-3.5-turbo-instruct

Memory: ConversationBufferMemory for chat history

Retrieval: Top-5 similar documents per query

🔧 Customization Options
Embedding Models
You can change the embedding model by modifying the SentenceTransformerEmbeddings initialization:

python
# In the RAGChatbot.__init__ method
self.embedding_model = SentenceTransformerEmbeddings("sentence-transformers/all-mpnet-base-v2")
Popular alternatives:

all-mpnet-base-v2: Higher quality, slower

all-distilroberta-v1: Good balance of speed and quality

paraphrase-multilingual-MiniLM-L12-v2: Multilingual support

Chunking Strategy
Modify chunking parameters in the chunk_text method:

python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increase for longer chunks
    chunk_overlap=100,  # Increase overlap for better context
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
LLM Configuration
Change the OpenAI model or parameters:

python
llm = OpenAI(
    temperature=0.3,  # Lower for more focused responses
    openai_api_key=openai_api_key,
    model_name="gpt-4"  # Use GPT-4 for better quality
)
🐛 Troubleshooting
Common Issues
"No module named 'sentence_transformers'"

bash
pip install sentence-transformers
ChromaDB permission errors

Ensure the application has write permissions in the current directory

Delete the chroma_db folder and restart the application

OpenAI API errors

Verify your API key is correct

Check your OpenAI account has sufficient credits

Ensure you're using a supported model

Memory issues with large documents

Reduce chunk size or process fewer documents at once

Consider using a smaller embedding model

Performance Optimization
For large document collections:

Use batch processing for document uploads

Consider using a more powerful embedding model

Implement document filtering based on metadata

For faster responses:

Reduce the number of retrieved documents (k parameter)

Use a smaller embedding model

Implement caching for frequent queries

📝 File Structure
text
rag-chatbot/
├── rag_chatbot_app.py      # Main application file
├── requirements.txt        # Python dependencies
├── .env.template          # Environment variables template
├── .env                   # Your environment variables (create this)
├── README.md              # This file
├── chroma_db/             # ChromaDB persistent storage (auto-created)
└── .streamlit/            # Streamlit configuration (optional)
    └── secrets.toml       # For deployment secrets
🤝 Contributing
Fork the repository

Create a feature branch: git checkout -b feature-name

Make your changes and test thoroughly

Commit your changes: git commit -m 'Add feature-name'

Push to the branch: git push origin feature-name

Create a Pull Request

🙏 Acknowledgments
LangChain: For the excellent RAG framework

ChromaDB: For the efficient vector database

Sentence-Transformers: For high-quality embeddings

Streamlit: For the beautiful and easy-to-use interface

OpenAI: For the powerful language models

Happy chatting with your documents! 🤖📚