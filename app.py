"""Sephra: Streamlit app (product showcase + retrieval chatbot).

Run from the project root:
    .venv\\Scripts\\streamlit.exe run app.py
"""

from __future__ import annotations

import streamlit as st

from sephra.generation.answer_generator import generate_answer
from sephra.generation.llm_client import LLMError, get_llm_client
from sephra.logging_config import setup_logging
from sephra.query.filters import apply_hard_filters
from sephra.query.llm_parser import LLMQueryParser
from sephra.query.rule_parser import parse_query
from sephra.query.schema import CategoryFilter, StructuredQuery
from sephra.retrieval.pipeline import DATASET_DISCLAIMER, SephraRetriever
from sephra.ui.components import CSS, product_header, render_response

st.set_page_config(page_title="Sephra", page_icon="💄", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
setup_logging()


@st.cache_resource(show_spinner="Loading catalogue, search index and embedding model (first run only)…")
def load_retriever() -> SephraRetriever:
    retriever = SephraRetriever()
    retriever.search("warm up moisturizer")   # loads models and builds the BM25 index once
    return retriever


@st.cache_resource
def load_llm(_retriever: SephraRetriever):
    """Return (client, parser, problem). problem explains why no LLM is available."""
    try:
        client = get_llm_client()
    except LLMError as exc:
        return None, None, str(exc)
    if client is None:
        return None, None, "No LLM configured in .env (retrieval-only mode)."
    return client, LLMQueryParser(client, _retriever.catalog), None


retriever = load_retriever()
catalog = retriever.catalog
llm_client, llm_parser, llm_problem = load_llm(retriever)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("💄 Sephra")
    st.caption("Beauty-product discovery with hybrid retrieval over customer reviews.")
    mode_label = st.radio("Retrieval mode", ["Hybrid (default)", "Dense only", "Hybrid + reranker (slow)"],
                          help="Hybrid = semantic + keyword search fused with RRF. The reranker showed no "
                               "measurable benefit in evaluation and adds 1–10 s per query.")
    mode = {"Hybrid (default)": "hybrid", "Dense only": "dense",
            "Hybrid + reranker (slow)": "hybrid_rerank"}[mode_label]
    top_k = st.slider("Products per answer", 3, 8, 5)
    st.divider()
    if llm_client is not None:
        use_llm = st.toggle("Use LLM (query understanding + answers)", value=True,
                            help="Free OpenRouter model. Every LLM output is validated; if the LLM "
                                 "fails or is rate-limited, Sephra falls back to rule-based parsing "
                                 "and still shows the retrieved evidence.")
        st.caption(f"Model: `{llm_client.model}` (free tier, rate-limited)")
    else:
        use_llm = False
        st.markdown(f"**LLM:** off. {llm_problem}")
    st.caption(DATASET_DISCLAIMER)
    st.caption(f"Loaded: {len(catalog.products):,} products · {len(catalog.reviews):,} reviews "
               f"({sum(1 for p in catalog.products.values() if p.loaded_review_count):,} products have "
               "reviews in the loaded partition).")

chat_tab, browse_tab = st.tabs(["💬 Ask Sephra", "🛍️ Browse products"])

# --------------------------------------------------------------------- chat
with chat_tab:
    st.markdown("Ask in plain English, e.g. *“moisturizer for dry skin under $30, not greasy”*, "
                "*“compare X and Y”* or *“what do customers say about …?”*")
    if "history" not in st.session_state:
        st.session_state.history = []

    for index, (question, response, answer) in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            render_response(response, retriever, key=f"h{index}", answer=answer)

    question = st.chat_input("What are you looking for?")
    if question:
        answer = None
        with st.spinner("Understanding your request…"):
            parser = llm_parser if use_llm else (lambda text, brands: parse_query(text, brands))
            structured = parser(question, retriever.known_brands)
        with st.spinner("Searching…"):
            response = retriever.search_structured(structured, mode=mode, top_k=top_k)
        if use_llm:
            with st.spinner("Writing a grounded answer…"):
                answer = generate_answer(response, llm_client)
        st.session_state.history.append((question, response, answer))
        st.rerun()

# ------------------------------------------------------------------- browse
with browse_tab:
    st.markdown("Filter the dataset catalogue directly. All filters are applied exactly.")
    products = list(catalog.products.values())
    c1, c2, c3 = st.columns(3)
    primary = c1.selectbox("Category", ["All"] + sorted({p.primary_category for p in products}))
    secondary_options = sorted({p.secondary_category for p in products
                                if p.secondary_category and (primary == "All" or p.primary_category == primary)})
    secondary = c2.selectbox("Subcategory", ["All"] + secondary_options)
    brand = c3.selectbox("Brand", ["All"] + sorted({p.brand for p in products}, key=str.lower))
    c4, c5, c6, c7 = st.columns(4)
    max_price = c4.number_input("Max price ($)", min_value=0.0, value=0.0, step=5.0,
                                help="0 = no limit")
    min_rating = c5.slider("Min rating", 0.0, 5.0, 0.0, 0.5)
    fragrance_free = c6.checkbox("Fragrance-free")
    in_stock = c7.checkbox("In stock only")
    sort_by = st.radio("Sort by", ["Rating", "Price (low to high)", "Reviews in dataset"], horizontal=True)

    categories = []
    if primary != "All" or secondary != "All":
        categories = [CategoryFilter(primary=None if primary == "All" else primary,
                                     secondary=None if secondary == "All" else secondary)]
    browse_query = StructuredQuery(
        original_query="browse", semantic_query="browse", categories=categories,
        brands=[] if brand == "All" else [brand], max_price=max_price or None,
        min_rating=min_rating or None, in_stock_only=in_stock,
        required_attributes=["Fragrance Free"] if fragrance_free else [])
    eligible = [catalog.products[pid] for pid in apply_hard_filters(browse_query, catalog).eligible_ids]
    key = {"Rating": lambda p: -(p.rating or 0), "Price (low to high)": lambda p: p.price_usd,
           "Reviews in dataset": lambda p: -p.loaded_review_count}[sort_by]
    eligible.sort(key=lambda p: (key(p), p.name))

    page_size = 12
    pages = max(1, (len(eligible) + page_size - 1) // page_size)
    page = st.number_input(f"Page (of {pages})", 1, pages, 1)
    st.caption(f"{len(eligible):,} matching products")
    columns = st.columns(2)
    for i, product in enumerate(eligible[(page - 1) * page_size: page * page_size]):
        with columns[i % 2], st.container(border=True):
            product_header(product)
