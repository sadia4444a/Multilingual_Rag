# Multilingual_Rag

Setup guide: 
🚀 Step-by-Step Setup
1. Clone the Repository
git clone git@github.com:sadia4444a/Multilingual_Rag.git
cd Multilingual_Rag


2. Install Poetry (if not installed)

For Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -
brew install poetry

 Or for Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -


3. Install project dependencies:
poetry install

4. Setup Environment Variables:

OPENAI_API_KEY=your-api-key

5. Run the Project
poetry run streamlit run app.py

Or

poetry shell
streamlit run app.py


full process in rag.py file

# 🌐 Multilingual_Rag

A Multilingual Retrieval-Augmented Generation (RAG) system for interactive question answering based on Rabindranath Tagore’s short story **“অপরিচিতা (Aparichita)”**.

Ask questions in **Bangla** or **English**, and get accurate answers generated directly from the story using OpenAI and LangChain.

---

## 📺 Demo Video

🔗 [Watch the Streamlit app demo](https://github.com/user-attachments/assets/db02ef0d-3e95-40ca-af4d-11c288ad3cff)

https://github.com/user-attachments/assets/3f690fe6-b795-4c4c-a96b-8d90f1b51868

<img width="1680" height="1050" alt="Screenshot 2025-07-26 at 7 20 15 PM" src="https://github.com/user-attachments/assets/c627d8af-94e6-411f-987a-de53d398196a" />




