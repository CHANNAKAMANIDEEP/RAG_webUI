import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import BitsAndBytesConfig
from huggingface_hub import login
import torch
import os
# Initialize session state variables for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'pdf_list' not in st.session_state:
    st.session_state['pdf_list'] = []
# Title of the app
st.title("RAG Chatbot with PDF Upload")
# Hugging Face login using environment variable or Streamlit secrets
hf_token = "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"
if not hf_token:
    hf_token = st.secrets["general"]["HUGGINGFACE_TOKEN"]  # Optional: Streamlit secrets
if hf_token:
    login(token=hf_token)
    st.success("Logged into Hugging Face successfully!")
else:
    st.error("Hugging Face token is missing. Please set the token in environment variables or secrets.")
# Sidebar section for uploading PDF and chat history
with st.sidebar:
    st.subheader("Uploaded PDFs")
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded_file:
        # Save the uploaded PDF to a temp directory
        if not os.path.exists("temp_pdfs"):
            os.makedirs("temp_pdfs")
        pdf_path = os.path.join("temp_pdfs", uploaded_file.name)
        # Write the uploaded file to the directory
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Append to the session state list of PDFs
        if uploaded_file.name not in st.session_state['pdf_list']:
            st.session_state['pdf_list'].append(uploaded_file.name)
    # Show the list of uploaded PDFs
    for pdf_name in st.session_state['pdf_list']:
        st.write(pdf_name)
    # Chat history
    st.subheader("Chat History")
    for i, chat in enumerate(st.session_state['chat_history']):
        st.write(f"Chat {i+1}: {chat['query']}")
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state['chat_history'].clear()
        st.rerun()
# Main section for chat interaction
st.subheader("Ask a question to the bot")
# User input
user_input = st.text_input("Enter your query:")
# If the user enters input and clicks send
if st.button("Send") and user_input:
    # If a PDF is uploaded, process it
    if uploaded_file:
        documents = SimpleDirectoryReader("temp_pdfs").load_data()
        # Embedding model
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Create an index from the uploaded documents
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, show_progress=True)
        # Define system prompt and query wrapper
        system_prompt = """
        You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided.
        """
        query_wrapper_prompt = "<|USER|>{query_str}<|ASSISTANT|>"
        # BitsAndBytes configuration for quantization (optional)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        # Set up the LLM
        llm = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.5, "do_sample": True},
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
            model_name="meta-llama/Llama-2-7b-chat-hf",
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto"
        )
        # Query the index
        query_engine = index.as_query_engine(llm=llm, streaming=False)
        response = query_engine.query(user_input)
        # Display chat interaction
        st.write(f"**User:** {user_input}")
        st.write(f"**Bot:** {response}")
        # Append to chat history in session state
        st.session_state['chat_history'].append({"query": user_input, "response": str(response)})
    else:
        st.write("Please upload a PDF to continue.")
# Display previous chat conversations
if st.session_state['chat_history']:
    st.subheader("Previous Chat")
    for chat in st.session_state['chat_history']:
        st.write(f"**User:** {chat['query']}")
        st.write(f"**Bot:** {chat['response']}")