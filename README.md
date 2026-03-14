# Conversational RAG Document Q&A App

A Streamlit-based web application that allows users to upload PDF documents and interact with their content through a conversational chat interface. The application uses Retrieval-Augmented Generation (RAG) to provide an accurate, context-aware question-answering experience, and it retains chat history to support follow-up questions in a natural conversation flow.

## Features

- **PDF Document Upload**: Upload one or multiple PDF files to be processed and queried.
- **Conversational Interface**: Chat with your documents using an intuitive Streamlit interface.
- **Context-Aware Memory**: The system maintains the chat history within a session, rewriting user queries to be self-contained based on the previous conversation.
- **Powered by Advanced LLMs**: Utilizes the `llama-3.3-70b-versatile` model via the **Groq API** for fast and high-quality generation.
- **Local Embeddings & Vector Store**: Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) and Chroma DB for efficient, local document chunk storage and retrieval.

## Technology Stack

- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **LLM Framework**: [LangChain](https://www.langchain.com/)
- **Language Model**: [Groq](https://groq.com/) (`llama-3.3-70b-versatile`)
- **Embeddings**: [HuggingFace](https://huggingface.co/) (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database**: [Chroma](https://www.trychroma.com/)
- **Document Parsing**: `pypdf`

## Setup & Installation

Follow these steps to run the application locally.

### 1. Clone the Repository
Navigate to your project directory.

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 3. Install Requirements
Install the necessary Python packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
The application requires certain environment variables. Create a `.env` file in the root directory and add your HuggingFace token if required by the embeddings model (the open-source model used doesn't strictly enforce it for local usage, but it is supported in the code).

```env
HF_TOKEN=your_huggingface_token_here
```
*Note: Your Groq API key will be entered directly into the Streamlit UI.*

## How to Run the Application

1. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```
2. Open the provided local URL (usually `http://localhost:8501`) in your web browser.
3. Enter your **Groq API Key** in the sidebar.
4. Specify a **Session ID** (or leave the default) to track your chat history.
5. Upload one or more PDF files.
6. Once the documents are processed and embedded, start asking questions in the chat!

## Project Structure

- `app.py`: The main Streamlit application file containing the UI logic and the RAG pipeline.
- `requirements.txt`: List of Python dependencies required to run the project.
- `.env`: Environment variables file (not included in version control) for API keys.
- `temp.pdf`: Temporary storage file used during the PDF upload and extraction process.
