import streamlit as st
import os
import json
import uuid
from base64 import b64decode
import re
import logging 
from langdetect import detect
from typing import Any, Optional
from typing import Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.storage import LocalFileStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMListwiseRerank
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.base import Docstore, AddableMixin
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.docstore.base import Docstore, AddableMixin
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMListwiseRerank
from langchain_text_splitters import RecursiveCharacterTextSplitter



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
# ===================== Paths & Setup =====================
FAISS_DIR = "./project_rag_24_jul/"
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "faiss_index")
DOCSTORE_DIR = os.path.join(FAISS_DIR, "docstore")
id_key = "doc_id"
# ========== Language Detection ==========
def detect_lang(text):
    try:
        return detect(text)
    except:
        return "en"

# ========== Optimize Query ==========
def optimize_query(query: str) -> dict:
    system_prompt = """
You are an expert in multilingual query optimization for a RAG (Retrieval-Augmented Generation) pipeline that supports both English and Bengali (Bangla).

Your job is to:
1. Reformulate the original question clearly and naturally in its own language (English or Bengali) for use with a vector-based retriever.
2. Extract concise and essential keywords (nouns, names, concepts) for use with a BM25 retriever. Only include important terms, and return them as a space-separated string — do not translate the language.

Output JSON must follow this format:
{
  "vector_query": "rephrased full question (same language)"
}
Only return valid JSON.
"""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt.strip()),
        HumanMessage(content=f"Original Query: {query}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"query": query})
    try:
        result = json.loads(response)
        return {
            "vector_query": result.get("vector_query", query).strip(),
        }
    except json.JSONDecodeError:
        st.warning("⚠️ LLM didn't return valid JSON. Using original query.")
        return {
            "vector_query": query,
        }
        

class DocumentLocalFileStore(LocalFileStore):
    def mset(self, key_value_pairs):
        serialized_pairs = [
            (key, json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}).encode('utf-8'))
            for key, doc in key_value_pairs
        ]
        super().mset(serialized_pairs)

    def mget(self, keys):
        byte_values = super().mget(keys)
        documents = []
        for value in byte_values:
            if value is None:
                documents.append(None)
            else:
                data = json.loads(value.decode('utf-8'))
                documents.append(Document(page_content=data["page_content"], metadata=data["metadata"]))
        return documents
class PersistentDocstore:
    def __init__(self, file_store: DocumentLocalFileStore):
        self.file_store = file_store

    def add(self, texts):
        self.file_store.mset(list(texts.items()))

    def delete(self, ids):
        for id in ids:
            self.file_store.delete(id)

    def search(self, search):
        result = self.file_store.mget([search])
        if result[0] is None:
            return f"Document with ID {search} not found."
        return result[0]

    def mget(self, keys):
        return self.file_store.mget(keys)

    def mset(self, key_value_pairs):
        self.file_store.mset(key_value_pairs)


# ========== Load Existing Docstore ==========
docstore = DocumentLocalFileStore(DOCSTORE_DIR)

# ========== Embeddings ==========
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# ========== Load Existing Vectorstore ==========
vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
)

# ========== MultiVectorRetriever ==========
multi_vector_retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key=id_key,
)

# ========== ParentDocumentRetriever ==========
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

pd_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    parent_splitter=parent_splitter,
    child_splitter=child_splitter,
    id_key=id_key,
)

# ========== Prompt Construction ==========
def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    lang = detect_lang(user_question)

    context_text = ""
    for text_element in docs_by_type["texts"]:
        source = text_element.metadata.get("source", "unknown")
        context_text += f"[Source: {source}]\n{text_element.page_content}\n\n"

    prompt_template = f"""
You are an intelligent assistant answering questions in Bengali or English based only on the provided context.

Instructions:
- Give short and direct answers.
- Do NOT explain unless absolutely necessary.
- Focus on accuracy — especially for names, numbers, and definitions.
- Use only the context below — no assumptions, no hallucination.
- Answer in the same language as the question ("{lang}").

Context:
{context_text}

Question:
{user_question}

Answer:
"""
    return ChatPromptTemplate.from_messages([
        HumanMessage(content=prompt_template.strip())
    ])

# ========== Combine Retrievers ==========
def combine_retrievers(query):
    optimized = optimize_query(query)
    vector_query = optimized["vector_query"]

    docs1 = multi_vector_retriever.invoke(vector_query)
    docs2 = pd_retriever.invoke(vector_query)

    seen = set()
    combined = []
    for doc in docs1 + docs2:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)
    return combined

# ========== Parse Docs ==========
def parse_docs(docs):
    b64 = []
    text = []
    for doc in docs:
        if not isinstance(doc, Document):
            continue
        try:
            b64decode(doc.page_content)
            b64.append(doc.page_content)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

# ========== Full RAG Chain ==========
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

chain = (
    {
        "context": RunnableLambda(combine_retrievers) | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | llm
    | StrOutputParser()
)


# ===================== Streamlit UI =====================
import streamlit as st

st.set_page_config(page_title="Aparichita QA", page_icon="📖")

st.title("📖 অপরিচিতা - প্রশ্নোত্তর ব্যবস্থা")
st.markdown("""
এই অ্যাপে আপনি **রবীন্দ্রনাথ ঠাকুরের** লেখা _“অপরিচিতা”_ গল্পের উপর ভিত্তি করে প্রশ্ন করতে পারবেন।  
বাংলা বা ইংরেজিতে প্রশ্ন করুন, এবং গল্পের প্রসঙ্গ থেকে উত্তর পাবেন।
""")

# গল্প পরিচিতি
with st.expander("📘 গল্পের সংক্ষিপ্ত পরিচিতি", expanded=True):
    st.write("""
“অপরিচিতা” প্রথম প্রকাশিত হয় প্রমথ চৌধুরী সম্পাদিত মাসিক “সবুজপত্র” পত্রিকার ১৩২১ বঙ্গাব্দের (১৯১৪) কার্তিক সংখ্যায়।  
এটি প্রথম গ্রন্থভুক্ত হয় 'গল্পসপ্তক'-এ এবং পরে “গল্পগুচ্ছ” তৃতীয় খণ্ডে (১৯২৭)।  
গল্পে কল্যাণী নামের এক বলিষ্ঠ নারীর চরিত্রের মাধ্যমে যৌতুক প্রথার বিরুদ্ধে নারী ও পুরুষের সম্মিলিত প্রতিবাদ ফুটে উঠেছে।
""")

# স্টেট ম্যানেজমেন্ট
if "history" not in st.session_state:
    st.session_state.history = []

if "current_qa" not in st.session_state:
    st.session_state.current_qa = None

# ইউজার ইনপুট
st.markdown("### ❓ আপনার প্রশ্ন লিখুন:")
query = st.text_input("প্রশ্ন লিখুন এখানে:", placeholder="উদাহরণ: কল্যাণীর চরিত্র কেমন ছিল?")

if st.button("🔍 উত্তর দেখুন") and query.strip():
    with st.spinner("✍️ উত্তর তৈরি হচ্ছে..."):
        try:
            answer = chain.invoke(query.strip())

            # আগের current_qa কে history-তে পাঠাও
            if st.session_state.current_qa:
                st.session_state.history.append(st.session_state.current_qa)

            # নতুন current_qa সেট করো
            st.session_state.current_qa = {"question": query.strip(), "answer": answer}

        except Exception as e:
            st.error(f"⚠️ ত্রুটি: {e}")

# বর্তমান প্রশ্ন ও উত্তর দেখানো
if st.session_state.current_qa:
    st.markdown("### ✅ বর্তমান প্রশ্ন ও উত্তর")
    qa = st.session_state.current_qa
    st.markdown(
        f"""
        <div style="background:#e7f7ec; padding:15px; border-radius:8px; margin-bottom:15px;">
            <p><b>প্রশ্ন:</b> {qa['question']}</p>
            <p><b>উত্তর:</b><br>{qa['answer']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# পূর্ববর্তী প্রশ্ন ও উত্তর দেখানো
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🗂️ পূর্ববর্তী প্রশ্ন ও উত্তর")
    for i, qa in enumerate(reversed(st.session_state.history)):
        st.markdown(
            f"""
            <div style="background:#f0f2f6; padding:15px; border-radius:8px; margin-bottom:15px;">
                <p><b>প্রশ্ন {len(st.session_state.history) - i}:</b> {qa['question']}</p>
                <p><b>উত্তর:</b><br>{qa['answer']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
