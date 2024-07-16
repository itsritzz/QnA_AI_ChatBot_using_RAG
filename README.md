# Project Title: Retrieval-Augmented Generation (RAG) using LangChain

## Overview

This project implements a Retrieval-Augmented Generation (RAG) application using LangChain. The application uses a combination of language models and retrievers to generate responses based on the context of the conversation and additional external data retrieved from specified web links.

## Features

- Integrates multiple language models and embeddings.
- Uses history-aware retrievers for context understanding.
- Implements a conversational chain with memory capabilities.
- Web-based user interface with Streamlit.
- Customizable and extendable for various applications.

## Requirements

- Python 3.10
- Streamlit
- dotenv
- LangChain
- Chroma
- HuggingFace Embeddings
- Groq API Key
- LangChain API Key

## Installation

```bash
# Clone the repository
git clone https://github.com/your-github-username/Langchain_RAG.git

# Navigate to the project directory
cd Langchain_RAG

# Create a virtual environment
python -m venv env

# Activate the virtual environment
# For Windows
.\env\Scripts\activate
# For macOS/Linux
source env/bin/activate

# Install the required packages
pip install -r requirements.txt

# Set up environment variables
touch .env
echo "GROQ_API_KEY=your_groq_api_key" >> .env
echo "LANGCHAIN_API_KEY=your_langchain_api_key" >> .env
