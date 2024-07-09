import PyPDF2
import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer



import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN" )


def extract_texts_from_pdfs(directory):
    text_data=[]
    pdf_files =[f for f in os.listdir(directory) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        with open(os.path.join(directory, pdf_file),'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text_data.append(page.extract_text())

    return text_data

directory= 'DATA'
text_data= extract_texts_from_pdfs(directory)

text_splitter= RecursiveCharacterTextSplitter(chunk_size=512)
documents= [text_splitter.split_text(text) for text in text_data]

documents= [item for sublist in documents for item in sublist]


embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store= FAISS.from_texts(documents, embedding=embeddings)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained("./flan-t5-base_tokenizer")
model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-base_model")



def  generate_response(prompt, vector_store, model, tokenizer):
    docs= vector_store.similarity_search(prompt, k=5)
    context= " ".join([doc.page_content for doc in docs])

    input_text= f"Conext: {context}\n\nQuestion: {prompt}\nAnswer:"

    inputs=tokenizer.encode(input_text, padding=True, max_length=512, truncation=True, return_tensors='pt')
    outputs= model.generate(inputs, max_length=150, num_return_sequences=1)
    answer= tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer

st.title("RAG chatbot")

prompt = st.text_input("Ask me anything:")

if prompt:
    response= generate_response(prompt, vector_store, model, tokenizer)
    st.write("Response:", response)
