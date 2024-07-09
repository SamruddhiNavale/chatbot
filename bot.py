import os
import fitz
import torch
import numpy as np
import faiss
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationRetrievalChain
from langchain.prompts import PromptTemplate
from langchain import LLMPredictor

# Setup paths and constants
pdf_directory = 'DATA/'
model_name = "google/flan-t5-xl"
dimension = 768  # Dimension of the embeddings from the model

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Initialize FAISS index
index = faiss.IndexFlatL2(dimension)

# Helper function to embed text
def embed_text(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model.get_encoder()(inputs.input_ids)[0]
    return embeddings.mean(dim=1).cpu().numpy()

# Function to parse PDFs and extract text with metadata
def parse_pdfs(directory):
    documents = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(directory, file_name)
            pdf_document = fitz.open(file_path)
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text = page.get_text()
                documents.append({
                    'text': text,
                    'metadata': {
                        'file_name': file_name,
                        'page_number': page_num + 1
                    }
                })
    return documents

# Parse the PDFs
documents = parse_pdfs(pdf_directory)
texts = [doc['text'] for doc in documents]
metadata = [doc['metadata'] for doc in documents]

# Embed the texts and add to FAISS index
embeddings = embed_text(texts)
index.add_with_ids(embeddings, np.arange(len(embeddings)))

# Custom predictor with metadata
class CustomPredictorWithMetadata(LLMPredictor):
    def __init__(self, model, tokenizer, index, metadata):
        self.model = model
        self.tokenizer = tokenizer
        self.index = index
        self.metadata = metadata

    def predict(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs.input_ids)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Retrieve relevant metadata
        query_embedding = embed_text([prompt])[0]
        D, I = self.index.search(np.array([query_embedding]), k=1)
        matched_metadata = self.metadata[I[0][0]]

        # Include metadata in response
        response_with_metadata = f"{response}\n\nSource: {matched_metadata['file_name']} (Page {matched_metadata['page_number']})"
        return response_with_metadata

llm_predictor = CustomPredictorWithMetadata(model, tokenizer, index, metadata)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Conversation retrieval chain
retrieval_chain = ConversationRetrievalChain(
    memory=memory,
    retriever=index,
    llm_predictor=llm_predictor,
    prompt_template=PromptTemplate("Answer the question based on the context:\n{context}\nQuestion: {question}\nAnswer:")
)

# Streamlit interface
st.title("RAG Chatbot with PDF Source")

if 'history' not in st.session_state:
    st.session_state['history'] = []

user_input = st.text_input("You: ")

if user_input:
    st.session_state['history'].append(f"You: {user_input}")
    context = "\n".join(st.session_state['history'])
    
    response = retrieval_chain({"question": user_input, "context": context})
    st.session_state['history'].append(f"Bot: {response}")

for message in st.session_state['history']:
    st.write(message)
