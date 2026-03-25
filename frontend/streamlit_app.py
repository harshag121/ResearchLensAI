import streamlit as st
import requests
import json
import pandas as pd
import os
from typing import Dict, Any

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="ResearchLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def get_stats():
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"total_papers": 0, "dimension": 384, "index_name": "unknown"}


def backend_is_available() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except Exception:
        return False

def search_papers(query: str, filters: Dict[str, Any] = None, limit: int = 10):
    try:
        payload = {
            "query": query,
            "limit": limit,
            "filters": filters
        }
        response = requests.post(f"{API_URL}/search", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: {response.text}")
            return []
    except Exception as e:
        st.error(f"Connection error: {e}")
        return []

def ingest_paper(title, abstract, authors, field, year, url):
    try:
        payload = {
            "title": title,
            "abstract": abstract,
            "authors": [a.strip() for a in authors.split(",")],
            "field": field,
            "year": int(year),
            "url": url
        }
        response = requests.post(f"{API_URL}/ingest", json=payload)
        if response.status_code == 200:
            return True
        else:
            st.error(f"Ingestion failed: {response.text}")
            return False
    except Exception as e:
        st.error(f"Connection error: {e}")
        return False

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/search.png", width=100)
    st.title("ResearchLens AI")
    st.info("Intelligent Academic Research Assistant powered by Endee")
    
    st.markdown("---")
    backend_ready = backend_is_available()
    if not backend_ready:
        st.error("Backend is offline. Start FastAPI and Endee to enable search/ingest.")

    st.markdown("---")
    stats = get_stats()
    st.metric("Indexed Papers", stats.get("total_papers", 0))
    st.caption(f"Index: {stats.get('index_name', 'default')}")
    
    st.markdown("---")
    st.subheader("Add New Paper")
    with st.form("ingest_form"):
        new_title = st.text_input("Title")
        new_abstract = st.text_area("Abstract")
        new_authors = st.text_input("Authors (comma-separated)")
        new_field = st.selectbox("Field", ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.IR", "cs.DB", "bio.QM"])
        new_year = st.number_input("Year", 2000, 2030, 2024)
        new_url = st.text_input("URL (optional)")
        submitted = st.form_submit_button("Ingest Paper")
        
        if submitted:
            can_submit = True
            if not backend_ready:
                st.error("Cannot ingest while backend is offline.")
                can_submit = False
            if not new_authors.strip():
                st.warning("Please provide at least one author.")
                can_submit = False
            if can_submit and new_title and new_abstract:
                if ingest_paper(new_title, new_abstract, new_authors, new_field, new_year, new_url):
                    st.success("Paper ingested successfully!")
                    st.rerun()
            elif can_submit:
                st.warning("Please provide title and abstract.")

# Main specific
st.title("🔍 Semantic Research Search")

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("What are you looking for?", placeholder="e.g., 'transformers in computer vision' or 'rendering 3d'")

with col2:
    filter_year = st.selectbox("Filter by Year", ["All Time", "2024", "2023", "2022", "Older"])

# Search Logic
if query:
    if not backend_ready:
        st.warning("Search is unavailable because backend is offline.")
        st.stop()

    with st.spinner("Searching knowledge base..."):
        filters = {}
        if filter_year != "All Time":
            if filter_year == "Older":
                filters["year"] = {"$lt": 2022}
            else:
                filters["year"] = int(filter_year)
                
        results = search_papers(query, filters)
        
        if results:
            st.success(f"Found {len(results)} relevant papers")
            
            for res in results:
                with st.expander(f"📄 {res['title']} ({res['metadata'].get('year', 'N/A')}) - Score: {res['score']:.2f}"):
                    st.markdown(f"**Authors:** {', '.join(res['authors'])}")
                    st.markdown(f"**Field:** {res['metadata'].get('field', 'N/A')}")
                    st.markdown(f"**Abstract:** {res['abstract']}")
                    if res['url']:
                        st.markdown(f"[Read Paper]({res['url']})")
        else:
            st.info("No papers found matching your query.")
else:
    st.markdown("""
    ### Welcome to ResearchLens AI!
    
    This tool helps you explore academic papers using semantic search powered by Endee Vector Database.
    
    **Features:**
    - 🧠 **Hybrid Search**: Combines keyword matching with semantic understanding
    - 📊 **Metadata Filtering**: Filter by year, field, or citation count
    - 🚀 **Fast Retrieval**: Powered by HNSW indexing
    
    **Getting Started:**
    1. Enter a search query above
    2. Use the sidebar to add new papers to the index
    3. Explore the results!
    """)
