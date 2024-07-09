import PyPDF2
import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set environment variables for SSL and Hugging Face API token
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Function to extract text and metadata from PDF files
def extract_texts_from_pdfs(directory):
    text_data = []
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        with open(os.path.join(directory, pdf_file), 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                text_data.append({
                    'text': page.extract_text(),
                    'source': pdf_file,
                    'page': page_num + 1
                })
    return text_data

# Directory containing PDF files
directory = 'DATA'
text_data = extract_texts_from_pdfs(directory)

# Split text into manageable chunks and include metadata
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512)
documents = []
for item in text_data:
    chunks = text_splitter.split_text(item['text'])
    for chunk in chunks:
        documents.append({
            'text': chunk,
            'metadata': {
                'source': item['source'],
                'page': item['page']
            }
        })

# Create embeddings for text chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
texts = [doc['text'] for doc in documents]
metadata = [doc['metadata'] for doc in documents]
vectors = embeddings.embed_texts(texts)

# Create FAISS vector store with metadata
vector_store = FAISS(vectors=vectors, metadata=metadata)

# Load tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl", use_auth_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl", use_auth_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Initialize conversational memory
memory = ConversationBufferMemory()

# Initialize conversational retrieval chain
conversational_chain = ConversationalRetrievalChain(
    vector_store=vector_store,
    memory=memory,
    model=model,
    tokenizer=tokenizer
)

# Streamlit interface
st.title("RAG Chatbot with Memory")
prompt = st.text_input("Ask me anything:")

if prompt:
    # Use conversational chain to get response
    response = conversational_chain.ask(prompt)
    st.write("Response:", response)
