__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from htmlTemplates import css, bot_template, user_template
from langchain_groq import ChatGroq


# ----------------------------
# PDF text extractor
# ----------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


# ----------------------------
# Text splitter
# ----------------------------
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)


# ----------------------------
# Vectorstore with Chroma
# ----------------------------
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory=None  # in-memory
    )
    return vectorstore


# ----------------------------
# Conversation chain (Groq LLM)
# ----------------------------
def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192",
        temperature=0,
        max_tokens=512,
    )

    retriever = vectorstore.as_retriever()
    
    # Create a prompt for contextualized questions
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Create a prompt for answering questions
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\n\n{context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain


# ----------------------------
# Handle user input
# ----------------------------
def handle_userinput(user_question):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    response = st.session_state.conversation.invoke({
        "input": user_question,
        "chat_history": st.session_state.chat_history
    })
    
    # Add to chat history
    st.session_state.chat_history.append(("human", user_question))
    st.session_state.chat_history.append(("ai", response["answer"]))
    
    # Display chat history
    for i in range(0, len(st.session_state.chat_history), 2):
        if i < len(st.session_state.chat_history):
            st.write(
                user_template.replace("{{MSG}}", st.session_state.chat_history[i][1]),
                unsafe_allow_html=True,
            )
        if i + 1 < len(st.session_state.chat_history):
            st.write(
                bot_template.replace("{{MSG}}", st.session_state.chat_history[i + 1][1]),
                unsafe_allow_html=True,
            )


# ----------------------------
# Streamlit main
# ----------------------------
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Documents processed! You can now ask questions.")


if __name__ == "__main__":
    main()
