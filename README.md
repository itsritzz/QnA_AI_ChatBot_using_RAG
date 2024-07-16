# Retrieval-Augmented Generation (RAG) using LangChain

## Overview

This project implements a Retrieval-Augmented Generation (RAG) application using LangChain. The application leverages language models and retrievers to generate responses based on the conversation's context and additional external data retrieved from specified web links.

## Features

- **Integrates Multiple Language Models and Embeddings:**
  - Uses the ChatGroq model and HuggingFaceEmbeddings for language processing and embedding generation.

- **History-Aware Retrievers for Context Understanding:**
  - Implements a history-aware retriever to consider previous interactions in the conversation for generating relevant responses.

- **Conversational Chain with Memory Capabilities:**
  - Utilizes a conversational chain that maintains session-specific chat history to provide contextually appropriate answers.

- **Web-Based User Interface with Streamlit:**
  - Provides an interactive web interface for users to input web links and interact with the assistant.

- **Customizable and Extendable:**
  - The application is designed to be flexible, allowing easy customization and extension for various use cases.

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-github-username/Langchain_RAG.git
    cd Langchain_RAG
    ```

2. **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv env
    source env/bin/activate  # For macOS/Linux
    .\env\Scripts\activate  # For Windows
    ```

3. **Install Required Packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables:**

    Create a `.env` file and add the necessary API keys:

    ```
    GROQ_API_KEY=your_groq_api_key
    LANGCHAIN_API_KEY=your_langchain_api_key
    ```

## Usage

1. **Start the Streamlit Application:**

    ```bash
    streamlit run app.py
    ```

2. **Interact with the Application:**

    - Open the provided local URL in your browser.
    - Input a website link in the sidebar to load content for the conversational chain.
    - Start chatting with the assistant and receive responses based on the loaded content.

## Functional Components

### Model and Embeddings Setup

- **Language Model:** Utilizes ChatGroq model for generating responses.
- **Embeddings:** Employs HuggingFaceEmbeddings for embedding generation.

### Session State Management

- **Initialization:** Ensures session-specific states are initialized for chat history, web links, user input, and message processing status.

### Conversational Chain Initialization

- **Document Loading and Splitting:** Loads documents from specified web paths and splits them into manageable chunks.
- **Vector Store and Retriever:** Creates a vector store from the documents and sets up a retriever for fetching relevant information.
- **History-Aware Retriever:** Uses a prompt template to rephrase user queries considering the conversation history.
- **Question Answering Chain:** Constructs a chain that uses the retriever and the language model to generate concise answers based on retrieved context.

### User Interface

- **Custom Styling:** Applies custom CSS to enhance the appearance of the Streamlit app.
- **Input Handling:** Captures user input and processes it to generate responses.
- **Chat History Display:** Displays the ongoing conversation, maintaining a clear interaction flow.

## Customization

- **Styling:** Customize the app's appearance by modifying the CSS in the Streamlit markdown section.
- **Model and Embeddings:** Switch to different language models or embeddings as required by updating the respective sections in the code.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License

This project is licensed under the MIT License.

## Acknowledgments

- **LangChain Team:** For the library and comprehensive documentation.
- **Streamlit Team:** For providing an excellent framework for web interfaces.
- **HuggingFace:** For the robust embeddings and model resources.

## Contact

- **Author:** Ritesh Kumar Singh
- **Portfolio:** [itsritz.my.canva.site](https://itsritz.my.canva.site)
- **LinkedIn:** [Ritesh Kumar Singh](https://www.linkedin.com/in/ritesh001/)
- **GitHub:** [itsritzz](https://github.com/itsritzz)

---

2024 © Developed by Ritesh Kumar Singh. All rights reserved.
