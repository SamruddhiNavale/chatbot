from langchain.llms import LLaMAcpp
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Initialize the conversation memory
memory = ConversationBufferMemory()

# Define a chat prompt template
chat_prompt_template = ChatPromptTemplate(
    input_variables=["input", "history"],
    template="""
    {history}
    User: {input}
    Assistant:"""
)

# Initialize llama.cpp model
model_path = "path/to/llama-2-7b.gguf"
llama_model = LLaMAcpp(model_path=model_path)

def generate_response(prompt, vector_store, model, documents, metadata, memory):
    # Retrieve similar documents
    docs = vector_store.similarity_search(prompt, k=5)
    context_chunks = []
    for doc in docs:
        doc_text = doc.page_content
        doc_index = documents.index(doc_text)
        doc_metadata = metadata[doc_index]
        context_chunks.append(f"{doc_metadata['source']} (Page {doc_metadata['page_number']}): {doc_text}")
    
    context = " ".join(context_chunks)

    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vector_store.as_retriever(),
        memory=memory,
        prompt_template=chat_prompt_template,
        return_source_documents=True
    )
    response = chain.run({"input": prompt, "history": memory.load_memory()})
    memory.save_context({"input": prompt}, {"result": response['result']})
    
    # Include the source documents' metadata in the response
    source_docs_metadata = []
    for doc in response['source_documents']:
        doc_text = doc.page_content
        doc_index = documents.index(doc_text)
        doc_metadata = metadata[doc_index]
        source_docs_metadata.append(f"{doc_metadata['source']} (Page {doc_metadata['page_number']}): {doc_text}")

    response['result'] += "\n\nSources:\n" + "\n".join(source_docs_metadata)

    return response['result'], context

# Load the vector store
vector_store = FAISS.load("vector_store.faiss")

# Example interaction
prompts = ["What is the main topic of the documents?", "Can you provide more details on the second document?"]

for prompt in prompts:
    response, context = generate_response(prompt, vector_store, llama_model, documents, metadata, memory)
    print(f"Question: {prompt}\nAnswer: {response}\nContext: {context}\n")
