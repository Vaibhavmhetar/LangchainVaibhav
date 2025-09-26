# app.py – Local Tourism Guide Chatbot
# Same robust structure as your SIWS app, adapted for tourism use case.

import sys
import traceback

try:
    import streamlit as st
except Exception:
    print("Missing dependency: streamlit. Run: pip install streamlit")
    raise

st.set_page_config(page_title="Local Tourism Guide Chatbot", page_icon="🗺️", layout="centered")

st.title("🗺️ Local Tourism Guide Chatbot")
st.markdown("Ask about **places to visit, local food, hotels, culture, or travel routes**.")

# ---------------- Helper for safe imports ----------------
missing_pkgs = []

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document
    from langchain.chains import RetrievalQA
    from langchain_community.llms import HuggingFacePipeline
except Exception:
    missing_pkgs.append("langchain-community")

try:
    import langchain
except Exception:
    missing_pkgs.append("langchain")

try:
    from transformers import pipeline
except Exception:
    missing_pkgs.append("transformers")

try:
    import sentence_transformers  # noqa: F401
except Exception:
    missing_pkgs.append("sentence-transformers")

try:
    import faiss  # noqa: F401
except Exception:
    missing_pkgs.append("faiss-cpu")

try:
    import torch  # noqa: F401
except Exception:
    missing_pkgs.append("torch")

if missing_pkgs:
    st.error("Some Python packages required by this app are not installed.")
    st.write("Missing packages: `" + "`, `".join(missing_pkgs) + "`")
    st.write("Run this command to install them:")
    st.code("pip install " + " ".join(missing_pkgs))
    st.stop()

# ---------------- Imports after validation ----------------
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# ---------------- Tourism Dataset ----------------
tourism_data = [
    {"question": "Tell me about Gateway of India",
     "answer": "An iconic arch monument built in 1924 in Mumbai, facing the Arabian Sea. Popular for photography and ferry rides to Elephanta Caves."},

    {"question": "What are the top attractions in Mumbai?",
     "answer": "Gateway of India, Marine Drive, Juhu Beach, Siddhivinayak Temple, Elephanta Caves, and Chhatrapati Shivaji Maharaj Museum."},

    {"question": "Suggest local food in Mumbai",
     "answer": "Must try Vada Pav, Pav Bhaji, Bhel Puri, Sev Puri, Misal Pav, and seafood near Girgaum Chowpatty."},

    {"question": "What are budget hotels in Mumbai?",
     "answer": "Some budget-friendly options: Hotel Residency Fort, City Palace Hotel, Hotel Kumkum, and backpacker hostels in Colaba."},

    {"question": "Tell me about Marine Drive",
     "answer": "A 3.6 km-long promenade along the coast of Mumbai, also known as the Queen’s Necklace due to its night-time lights."},

    {"question": "Best time to visit Mumbai?",
     "answer": "November to February offers pleasant weather. Avoid June–September monsoon season due to heavy rains."},

    {"question": "How to travel within Mumbai?",
     "answer": "Options: Mumbai local trains (fastest), BEST buses, metro (limited), taxis, rickshaws, and app-based cabs like Ola/Uber."},

    {"question": "Tell me about Elephanta Caves",
     "answer": "UNESCO World Heritage site located on Elephanta Island. Famous for rock-cut temples dedicated to Lord Shiva, accessible by ferry from Gateway of India."},

    {"question": "Which festivals are famous in Mumbai?",
     "answer": "Ganesh Chaturthi, Diwali, Holi, Eid, and Kala Ghoda Arts Festival are major cultural events in Mumbai."},

    {"question": "What shopping places are in Mumbai?",
     "answer": "Colaba Causeway, Linking Road (Bandra), Crawford Market, and Fashion Street are famous shopping destinations."}
]

# ---------------- Prepare Documents ----------------
docs = [Document(page_content=item["answer"], metadata={"question": item["question"]}) for item in tourism_data]

# ---------------- Embeddings ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ---------------- FAISS Vector Store ----------------
try:
    vectorstore = FAISS.from_documents(docs, embeddings)
except Exception as e:
    st.error("Failed to create FAISS vectorstore. Ensure faiss-cpu is installed.")
    st.code(traceback.format_exc())
    st.stop()

# ---------------- QA Model ----------------
try:
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
except Exception:
    st.error("Failed to initialize HuggingFace pipeline. Check 'transformers' and 'torch' installation.")
    st.code(traceback.format_exc())
    st.stop()

# ---------------- QA Chain ----------------
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=True
    )
except Exception:
    st.error("Failed to build RetrievalQA chain.")
    st.code(traceback.format_exc())
    st.stop()

# ---------------- Streamlit UI ----------------
st.sidebar.title("📌 Quick Tourism FAQs")
quick_links = [
    "Top attractions in Mumbai",
    "Best local food",
    "Budget hotels",
    "Best time to visit",
    "Travel within city",
    "Famous festivals"
]
for item in quick_links:
    st.sidebar.markdown(f"- {item}")

query = st.text_input("💬 Ask about Mumbai tourism:")

if st.button("Get Answer") and query:
    with st.spinner("Searching for the best answer..."):
        try:
            result = qa_chain(query)
        except Exception:
            st.error("Error while running the QA chain.")
            st.code(traceback.format_exc())
            st.stop()

        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

    if source_docs:
        matched_question = source_docs[0].metadata.get("question", "")
        st.success(f"**Answer:** {answer}")
        st.info(f"📌 Based on: *{matched_question}*")
    else:
        st.warning("Sorry, I couldn't find an answer. Try rephrasing your question.")
