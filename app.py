import streamlit as st
import os

from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDFs and interact with their content")

api_key = st.text_input("Enter your Groq API Key:", type="password")

if api_key:

    llm = ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile"
    )

    session_id = st.text_input("Session ID", value="default_session")

    # Store chat history
    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:

        documents = []

        for uploaded_file in uploaded_files:

            temp_pdf = "./temp.pdf"

            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)

            docs = loader.load()

            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=session_id
        )

        retriever = vectorstore.as_retriever()

        # Question contextualization prompt
        contextualize_q_system_prompt = (
            "Given a conversation history and the latest user question, "
            "rewrite the user's question so it is self-contained."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # History aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )

        # QA prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Keep the answer concise.\n\n"
            "Context: {context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # QA chain
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        # RAG chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        # Session history function
        def get_session_history(session_id: str) -> BaseChatMessageHistory:

            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()

            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(

            rag_chain,

            get_session_history,

            input_messages_key="input",

            history_messages_key="chat_history",

            output_messages_key="answer"
        )

        user_input = st.text_input("Your Question:")

        if user_input:

            session_history = get_session_history(session_id)

            response = conversational_rag_chain.invoke(

                {"input": user_input},

                config={
                    "configurable": {"session_id": session_id}
                }
            )

            st.success(f"Assistant: {response['answer']}")

            st.write("Chat History:", session_history.messages)