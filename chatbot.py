import PyPDF2
import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
            'source': item['source'],
            'page': item['page']
        })

# Create embeddings for text chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embedding=embeddings, metadata_fields=['source', 'page'])

# Load tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl", use_auth_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl", use_auth_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Initialize conversational memory
memory = ConversationBufferMemory()

# Function to generate response from model with memory and metadata
def generate_response(prompt, vector_store, model, tokenizer, memory):
    # Retrieve memory
    past_conversation = memory.load_memory()
    
    # Get relevant documents from vector store
    docs = vector_store.similarity_search(prompt, k=5)
    context = " ".join([doc.page_content for doc in docs])
    
    # Create input with past conversation context
    input_text = f"Context: {context}\n\nConversation: {past_conversation}\n\nQuestion: {prompt}\nAnswer:"
    inputs = tokenizer.encode(input_text, padding=True, max_length=512, truncation=True, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Update memory with the latest prompt and response
    memory.save_memory(f"User: {prompt}\nBot: {answer}")
    
    # Include source information in the response
    source_info = "\nSources:\n" + "\n".join([f"{doc.metadata['source']} (Page {doc.metadata['page']})" for doc in docs])
    full_response = answer + source_info
    
    return full_response

# Streamlit interface
st.title("RAG Chatbot with Memory")
prompt = st.text_input("Ask me anything:")

if prompt:
    response = generate_response(prompt, vector_store, model, tokenizer, memory)
    st.write("Response:", response)
