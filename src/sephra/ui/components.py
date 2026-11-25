"""Reusable Streamlit rendering functions for products, reviews and results."""

from __future__ import annotations

import html

import streamlit as st

from sephra.catalog import Product
from sephra.retrieval.pipeline import ProductResult, SearchResponse
from sephra.retrieval.reviews import ReviewEvidence

SENTIMENT_LABEL = {"positive": "👍 Positive", "mixed": "😐 Mixed", "critical": "👎 Critical",
                   "unknown": "Review"}

CSS = """
<style>
.sephra-tag {display:inline-block; padding:2px 8px; margin:2px 4px 2px 0; border-radius:10px;
  font-size:0.78rem; background:rgba(150,110,160,0.15); border:1px solid rgba(150,110,160,0.35);}
.sephra-muted {color:rgba(120,120,120,1); font-size:0.85rem;}
.sephra-price {font-size:1.15rem; font-weight:600;}
</style>
"""


def tags(items: list[str]) -> str:
    return "".join(f'<span class="sephra-tag">{html.escape(i)}</span>' for i in items)


def rating_text(product: Product) -> str:
    return f"⭐ {product.rating:.2f}" if product.rating is not None else "⭐ no rating in dataset"


def product_header(product: Product, rank: int | None = None) -> None:
    prefix = f"{rank}. " if rank else ""
    st.markdown(f"#### {prefix}{html.escape(product.name)}")
    st.markdown(f'<span class="sephra-muted">{html.escape(product.brand)} · '
                f'{html.escape(product.category_path)} · {product.product_id}</span>',
                unsafe_allow_html=True)
    cols = st.columns(4)
    price = f"${product.price_usd:.2f}"
    if product.sale_price_usd:
        price += f" (sale ${product.sale_price_usd:.2f})"
    cols[0].markdown(f'<span class="sephra-price">{price}</span>', unsafe_allow_html=True)
    cols[1].markdown(rating_text(product))
    cols[2].markdown(f"💬 {product.loaded_review_count} reviews in our dataset")
    cols[3].markdown("🔴 Out of stock (snapshot)" if product.out_of_stock else "🟢 In stock (snapshot)")
    if product.highlights:
        st.markdown(tags(product.highlights), unsafe_allow_html=True)


def review_stats(evidence: ReviewEvidence) -> None:
    s = evidence.stats
    if not s:
        return
    rate = f", {s.recommend_rate:.0%} recommend" if s.recommend_rate is not None else ""
    st.markdown(f"**All {s.total} loaded reviews:** {s.positive} positive · {s.mixed} mixed · "
                f"{s.critical} critical{rate}")
    st.progress(s.positive / s.total, text=f"{s.positive / s.total:.0%} rated 4–5 stars")


def review_list(evidence: ReviewEvidence) -> None:
    for note in evidence.notes:
        st.info(note)
    if not evidence.selected:
        return
    st.caption(f"Showing {len(evidence.selected)} selected excerpts (chosen for relevance and to include "
               "different experiences). They are not a representative sample. See the totals above.")
    for selected in evidence.selected:
        review = selected.review
        skin = f" · reviewer skin: {review.skin_type}" if review.skin_type else ""
        recommend = {True: " · recommends", False: " · does not recommend"}.get(review.is_recommended, "")
        st.markdown(f"**{SENTIMENT_LABEL[review.sentiment]} — {review.rating}★**{skin}{recommend}  \n"
                    f"<span class='sephra-muted'>{html.escape(selected.reason)}</span>",
                    unsafe_allow_html=True)
        text = review.text if len(review.text) < 700 else review.text[:700] + "…"
        st.markdown(f"> {html.escape(text)}")


def why_matched(result: ProductResult, response: SearchResponse) -> list[str]:
    reasons = []
    constraints = response.query.describe_constraints()
    if constraints:
        reasons.append("Meets all your requirements: " + "; ".join(constraints))
    c = result.candidate
    if c:
        ranks = []
        if c.dense_rank:
            ranks.append(f"semantic rank #{c.dense_rank}")
        if c.bm25_rank:
            ranks.append(f"keyword rank #{c.bm25_rank}")
        if ranks:
            reasons.append("Retrieved by " + " and ".join(ranks))
        if c.matched_skin_tags:
            reasons.append("Listing tags match your skin type: " + ", ".join(c.matched_skin_tags))
    return reasons


def product_card(result: ProductResult, response: SearchResponse, retriever, key: str) -> None:
    product, evidence = result.product, result.evidence
    with st.container(border=True):
        product_header(product, result.rank)
        for reason in why_matched(result, response):
            st.markdown(f"- {reason}")
        if result.variants:
            st.markdown("**Related variants** (separate products, each with its own data): " + "; ".join(
                f"{v.name} ({v.product_id}, ${v.price_usd:.2f}, "
                f"{'rating ' + format(v.rating, '.2f') if v.rating is not None else 'no rating'}, "
                f"{v.loaded_review_count} reviews)" for v in result.variants))
        review_stats(evidence)
        with st.expander(f"Customer reviews ({len(evidence.selected)} selected of "
                         f"{evidence.loaded_review_count} in our dataset)"):
            review_list(evidence)
            if evidence.loaded_review_count:
                question = st.text_input("Ask about this product's reviews (e.g. scent, texture, oily skin)",
                                         key=f"q_{key}")
                if question:
                    follow_up = retriever.review_evidence_for(product.product_id, question)
                    st.markdown(f"**Reviews most relevant to “{html.escape(question)}”:**")
                    review_list(follow_up)
        if product.ingredients:
            with st.expander("Ingredients (from the dataset)"):
                st.write(product.ingredients)


def interpretation(response: SearchResponse) -> None:
    q = response.query
    with st.expander("How Sephra interpreted your request", expanded=False):
        st.markdown(f"**Intent:** {q.intent.value}  ·  **Parser:** {q.parser}")
        st.markdown("**Hard requirements (enforced):** "
                    + ("; ".join(q.describe_constraints()) or "none"))
        if q.skin_types:
            st.markdown("**Preferences (ranking only):** " + ", ".join(s.value + " skin" for s in q.skin_types))
        st.markdown(f"**Search text:** {q.semantic_query}")
        if response.filter_summary:
            st.markdown(f"**Filter funnel:** {response.filter_summary}")
        for note in q.notes:
            st.caption(note)
        if response.timings:
            st.caption("Timings: " + ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in response.timings.items()))


def comparison_table(response: SearchResponse) -> None:
    columns = st.columns(len(response.products))
    for column, result in zip(columns, response.products):
        p, s = result.product, result.evidence.stats
        with column:
            st.markdown(f"**{html.escape(p.name)}**  \n{html.escape(p.brand)} · {p.product_id}")
            st.markdown(f"- Price: ${p.price_usd:.2f}\n- {rating_text(p)}\n"
                        f"- Category: {p.category_path}\n- Size: {p.size or 'n/a'}\n"
                        f"- Reviews in dataset: {p.loaded_review_count}")
            if s:
                st.markdown(f"- Review split: {s.positive}👍 / {s.mixed}😐 / {s.critical}👎")
                if s.recommend_rate is not None:
                    st.markdown(f"- Recommend rate: {s.recommend_rate:.0%}")
            if p.highlights:
                st.markdown(tags(p.highlights), unsafe_allow_html=True)


def llm_answer(answer) -> None:
    """Show the generated answer, or explain clearly why there is none."""
    if answer is None:
        st.caption("🧪 Retrieval-only mode: LLM answers are switched off. "
                   "Below are the retrieved products and their evidence.")
        return
    if answer.status == "generated":
        with st.container(border=True):
            st.markdown(answer.text)
            st.caption(f"✍️ Written by {answer.model} using only the evidence below. "
                       f"✅ Passed automatic checks: every price, rating and product ID appears in the "
                       f"evidence ({answer.latency_s:.1f} s, attempt {answer.attempts}). "
                       "Customer opinions are quoted from reviews, not verified facts.")
    elif answer.status != "skipped":
        st.warning(f"No AI-written answer: {answer.message}")


def render_response(response: SearchResponse, retriever, key: str, answer=None) -> None:
    interpretation(response)
    if response.status != "ok":
        st.warning(response.message)
        return
    st.success(response.message)
    llm_answer(answer)
    st.markdown("##### Evidence")
    if response.query.intent.value == "product_comparison":
        comparison_table(response)
    for result in response.products:
        product_card(result, response, retriever, key=f"{key}_{result.product.product_id}")
