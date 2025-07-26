# Multilingual_Rag

Setup guide: 
🚀 Step-by-Step Setup
1. Clone the Repository
git clone git@github.com:sadia4444a/Multilingual_Rag.git
cd Multilingual_Rag


2. Install Poetry (if not installed)

# For Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -
brew install poetry

# Or for Windows (PowerShell)
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

---

## 🚀 Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone git@github.com:sadia4444a/Multilingual_Rag.git
cd Multilingual_Rag





