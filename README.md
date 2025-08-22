# 📚 Ask-My-PDF — Chat with Your Documents

**Ask-My-PDF** is a powerful conversational AI application that allows you to chat with your PDF documents. By leveraging cutting-edge LLMs from **Groq** and robust vector store technology from **ChromaDB**, the app enables you to ask questions about your documents and receive intelligent, context-aware answers. It's the perfect tool for extracting information and summarizing content from one or multiple PDF files.

***

### 🚀 Key Features

* **🧠 LLM-Powered Chat:** Integrates with the high-speed **Groq API** and **LLaMA 3** for rapid, accurate responses to your questions.

* **📄 Multiple PDF Support:** Allows you to upload and process several PDF documents simultaneously.

* **🔍 Semantic Search:** Uses a vector store (**ChromaDB**) and a lightweight embedding model to perform semantic searches and find the most relevant information in your documents.

* **📝 Automated Text Extraction:** Automatically extracts and chunks text from PDF files, preparing it for the language model.

* **💻 Clean UI:** Provides a simple and intuitive web interface built with **Streamlit**.

***

### 🧱 Architecture Overview

The system processes user queries by first retrieving relevant information from the uploaded documents and then using that context to generate a response with the LLM.
***
medicare.ai/
│
├── data/                          # Raw data files
│   └── doctor_availability.csv
│
├── notebook/                      # Exploratory notebooks
│   └── availability.csv
│
├── data_models/                   # Pydantic models and DB logic
│   ├── __init__.py
│   └── models.py
│
├── prompt_library/               # Prompt templates and logic
│   ├── __init__.py
│   └── prompt.py
│
├── toolkit/                      # Helper functions and tools
│   ├── __init__.py
│   └── toolkits.py
│
├── utils/
│   ├── __init__.py
│   └── llms.py                   # Groq + LLaMA LLM integration
│
├── agent.py                      # Agent definitions using LangGraph
├── main.py                       # Main entrypoint to build the graph
├── streamlit_ui.py               # Streamlit front-end
├── setup.py                      # Package setup
├── requirements.txt              # Dependencies
├── LICENSE
└── README.md                     # You are here!
'''
***

### ⚙️ Setup Instructions

To run this application locally, follow these simple steps.

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/rahuljat27/Chat_with_multiple_PDFs.git](https://github.com/rahuljat27/Chat_with_multiple_PDFs.git)
    cd Chat_with_multiple_PDFs
    ```

2.  **Install Requirements**
    First, create a virtual environment to manage your dependencies.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Configure Your API Key**
    Create a file named `.env` in the project's root directory and add your Groq API key.

    ```ini
    # .env
    GROQ_API_KEY="your_api_key_here"
    ```

4.  **Run the App**
    Start the Streamlit application from your terminal.

    ```bash
    streamlit run app.py
    ```

### 🌐 Live Application

Your application is deployed and live at:
[https://askmypdfs27.streamlit.app/](https://askmypdfs27.streamlit.app/)

***

### 🧪 Sample Use Cases

* **User:** "What are the key findings in these documents?"

* **User:** "Can you summarize the introduction of the first PDF?"

* **User:** "What is the capital of France?" (The model will respond based on general knowledge, not the PDFs.)

***

### ✅ To-Do

* [ ] Add chat history persistence (e.g., to a database).

* [ ] Allow users to save and load previous conversations.

* [ ] Integrate different LLMs for more model flexibility.

* [ ] Implement a `streamlit_app.toml` file to customize deployment settings.
