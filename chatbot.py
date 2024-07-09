import PyPDF2
import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setting environment variables
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def extract_texts_from_pdfs(directory):
    text_data = []
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        with open(os.path.join(directory, pdf_file), 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_number, page in enumerate(reader.pages, start=1):
                text_data.append({
                    'source': pdf_file,
                    'page_number': page_number,
                    'text': page.extract_text()
                })

    return text_data

def generate_response(prompt, vector_store, model, tokenizer, text_data):
    docs = vector_store.similarity_search(prompt, k=5)
    context = " ".join([f"{doc['source']} (Page {doc['page_number']}): {doc['text']}" for doc in docs])

    input_text = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"

    inputs = tokenizer.encode(input_text, padding=True, max_length=1024, truncation=True, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, context

# Example directory and data extraction
directory = 'DATA'
text_data = extract_texts_from_pdfs(directory)

# Initialize embeddings and vector store
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512)
documents = [text_splitter.split_text(entry['text']) for entry in text_data]
documents = [item for sublist in documents for item in sublist]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(documents, embedding=embeddings)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", use_auth_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", use_auth_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Streamlit app setup
st.title("RAG chatbot")
prompt = st.text_input("Ask me anything:")

if prompt:
    response, metadata = generate_response(prompt, vector_store, model, tokenizer, text_data)
    st.write("Response:", response)
    st.write("Metadata:", metadata)
