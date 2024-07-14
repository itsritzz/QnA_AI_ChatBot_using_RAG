import os

# Set the environment variable to use the pure-Python implementation of protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from dotenv import load_dotenv

# Initialize environment variables from Streamlit Secrets
os.environ["GROQ_API_KEY"] = os.getenv["GROQ_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = os.getenv["LANGCHAIN_API_KEY"]

# Import necessary components for the model
try:
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_chroma import Chroma
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.chat_history import BaseChatMessageHistory, HumanMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as e:
    st.error(f"An error occurred while importing modules: {e}")
    st.stop()

# Set up the model and embeddings
llm = ChatGroq(model="llama3-8b-8192")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Initialize session state keys
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}
if "web_link" not in st.session_state:
    st.session_state["web_link"] = ""
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
if "message_processed" not in st.session_state:
    st.session_state["message_processed"] = False

# Function to initialize the conversational chain
def initialize_conversational_chain(web_path):
    loader = WebBaseLoader(web_paths=(web_path,))
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.chat_history:
            st.session_state.chat_history[session_id] = ChatMessageHistory()
        return st.session_state.chat_history[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

# Streamlit UI
# Adding custom CSS for background and styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #f7f7f7, #e2e2e2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #003366, #004080);
    }
    .header {
        font-size: 36px;
        color: white;
        background-color: #004080;
        padding: 20px;
        text-align: center;
    }
    .footer {
        font-size: 12px;
        color: blue;
        padding: 20px;
        text-align: center;
        position: fixed;
        bottom: 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .footer a {
        color: white;
        margin: 10px;
        text-decoration: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="header">Retrieval-Augmented Generation (RAG) using LangChain</div>', unsafe_allow_html=True)

st.sidebar.header("Input a Website Link")
web_link = st.sidebar.text_input("Website URL")

if st.sidebar.button("Load"):
    st.session_state.web_link = web_link
    st.session_state.message_processed = False  # Reset message processed flag

if st.session_state.web_link:
    st.write(f"Loaded content from: {st.session_state.web_link}")
    conversational_rag_chain = initialize_conversational_chain(st.session_state.web_link)
    session_id = "user_session"

    st.write("Start chatting with the assistant (type 'exit' to stop):")

    # Form to capture user input
    with st.form(key="input_form"):
        user_input = st.text_input("You:")
        submit_button = st.form_submit_button(label="Send Message")

    # Check if the form is submitted
    if submit_button:
        st.session_state.user_input = user_input
        st.session_state.message_processed = False  # Allow processing

    # Process the input if it's not processed yet
    if not st.session_state.message_processed and st.session_state.user_input:
        if st.session_state.user_input.lower() == "exit":
            st.write("Conversation ended.")
        else:
            result = conversational_rag_chain.invoke(
                {"input": st.session_state.user_input},
                config={
                    "configurable": {"session_id": session_id}
                }
            )
            st.session_state.chat_history[session_id].add_user_message(st.session_state.user_input)
            st.session_state.chat_history[session_id].add_ai_message(result["answer"])
            st.session_state.message_processed = True
            st.session_state.user_input = ""  # Clear the input after processing
            st.experimental_rerun()  # Rerun the script to refresh the interface

# Display chat history
if "user_session" in st.session_state.chat_history:
    st.write("Chat History:")
    for message in st.session_state.chat_history["user_session"].messages:
        if isinstance(message, HumanMessage):
            st.write(f"You: {message.content}")
        elif isinstance(message, AIMessage):
            st.write(f"Assistant: {message.content}")

# Footer
st.markdown(
    '''
    <div class="footer">
        <div style="text-align: left;">
            2024 © Developed by Ritesh Kumar Singh. All rights reserved
            <br>
            &#160; &#160; &#160; &#160; &#160; &#160; &#160; &#160; &#160; &#160; &#160; &#160; &#160; &#160; &#160;
            <a href="https://itsritz.my.canva.site" target="_blank" style="color: blue;">Portfolio</a> |
            <a href="https://www.linkedin.com/in/ritesh001/" target="_blank" style="color: blue;">LinkedIn</a> | 
            <a href="https://github.com/itsritzz" target="_blank" style="color: blue;">GitHub</a> 
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)

# Reset the flag after rerun
st.session_state.message_processed = False
