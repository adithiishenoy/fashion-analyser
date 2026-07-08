import streamlit as st
from groq import Groq
import base64
import json
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fashion Image Analyzer", page_icon="👗", layout="wide")

# ==========================================
# PRODUCT MANAGEMENT & ARCHITECTURE SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/tags.png", width=60)
    st.title("Product Overview")
    st.caption("Designed & Developed by K Adithi Shenoy")
    
    st.markdown("---")
    st.markdown("### 🎯 Core Value Proposition")
    st.markdown(
        "Automating manual inventory onboarding workflows for fashion e-commerce platforms. "
        "By replacing manual tagging with multimodal LLM extraction, marketplaces can scale "
        "cataloging throughput while ensuring data consistency."
    )
    
    st.markdown("### 💼 Operational Use Cases")
    st.markdown(
        "- **Warehouse Intake:** Auto-generate e-commerce titles, descriptions, and metadata labels at scale.\n"
        "- **Demand Analytics:** Aggregate structured tags (colors/styles) across wholesale inventory shipments to track assortment trends.\n"
        "- **Faceted Search SEO:** Standardize taxonomy constraints to power accurate frontend filtering."
    )
    
    st.markdown("### 🛠️ Technical Architecture")
    st.markdown(
        "- **Orchestration:** Streamlit Engine\n"
        "- **Inference Layer:** Groq API Cloud\n"
        "- **Multimodal Engine:** Llama-4-Scout (17B)\n"
        "- **Analytics:** Pandas & Matplotlib batch processing"
    )
    st.markdown("---")

# ==========================================
# MAIN APPLICATION PAGE
# ==========================================
st.title("👗 Automated Fashion Cataloging & Tagging Engine")
st.markdown(
    "Upload one or multiple apparel product images to instantly generate standardized catalog tags, "
    "structured JSON payloads, and automated e-commerce descriptions."
)

# Expandable Product Logic section to show product thinking
with st.expander("💡 View Product Logic & Taxonomy Mapping Constraints", expanded=False):
    st.markdown("""
    * **Strict Output Typing:** Enforces consistent JSON parsing to feed into relational databases without schema drift.
    * **Creative Attribution:** Leverages zero-shot vision intelligence to generate natural-language descriptions matching standard marketing tones.
    * **Batch Analytics Ready:** Aggregates real-time metadata distributions for category and color assortment audits.
    """)

st.divider()

# API Key Validation
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    api_key = st.text_input("Enter your Groq API key", type="password", help="Your key is processed securely and never saved.")

if not api_key:
    st.info("ℹ️ Please provide a Groq API key above (or configure it via Streamlit secrets) to execute live inference.")
    st.stop()

client = Groq(api_key=api_key)

uploaded_files = st.file_uploader(
    "Upload product asset(s)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
)

PROMPT = """You are a fashion product analyst. Look at this clothing/fashion product image
and return ONLY a valid JSON object (no markdown, no extra text) with these exact keys:

{
  "category": "e.g. dress, t-shirt, sneakers, handbag",
  "primary_color": "main color",
  "secondary_colors": ["list", "of", "other", "colors"],
  "style": "e.g. casual, formal, streetwear, athleisure, bohemian",
  "occasion": "e.g. everyday, party, office, gym, festive",
  "target_demographic": "e.g. women, men, unisex, kids",
  "fabric_guess": "best guess at fabric/material",
  "description": "a 1-2 sentence product description suitable for an e-commerce listing"
}

Return ONLY the JSON object, nothing else."""

def analyze_image(image_bytes, media_type):
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{media_type};base64,{b64_image}"

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=500,
    )

    text_response = completion.choices[0].message.content.strip()

    if text_response.startswith("```"):
        text_response = text_response.strip("`")
        if text_response.startswith("json"):
            text_response = text_response[4:]

    return json.loads(text_response.strip())

if uploaded_files:
    results = []

    for uploaded_file in uploaded_files:
        st.divider()
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(uploaded_file, caption=f"Asset: {uploaded_file.name}", use_container_width=True)

        media_type = uploaded_file.type
        image_bytes = uploaded_file.getvalue()

        with col2:
            st.subheader("📋 Extracted Catalog Attributes")
            with st.spinner(f"Extracting metadata from {uploaded_file.name}..."):
                try:
                    result = analyze_image(image_bytes, media_type)
                    result["filename"] = uploaded_file.name
                    results.append(result)

                    # Presenting structured data cleanly using a structured layout
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.markdown(f"**Category:** `{result.get('category')}`")
                        st.markdown(f"**Primary Color:** `{result.get('primary_color')}`")
                        st.markdown(f"**Style Taxonomy:** `{result.get('style')}`")
                    with m_col2:
                        st.markdown(f"**Target Intent:** `{result.get('occasion')}`")
                        st.markdown(f"**Demographic:** `{result.get('target_demographic')}`")
                        st.markdown(f"**Material Profile:** `{result.get('fabric_guess')}`")
                    
                    st.markdown(f"**Secondary Colors:** {', '.join([f'`{c}`' for c in result.get('secondary_colors', [])])}")
                    st.info(f"**Generated Marketing Copy:**\n{result.get('description')}")

                except json.JSONDecodeError:
                    st.error("Failed to parse visual data into strict schema. Please re-run ingestion.")
                except Exception as e:
                    st.error(f"Execution Error: {e}")

    if len(results) > 1:
        st.divider()
        st.header("📊 Assortment Analytics & Batch Export")
        st.markdown("Aggregated analytical overview of processed warehouse batch assets for operational verification.")

        df = pd.DataFrame(results)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            df["category"].value_counts().plot(kind="bar", ax=ax, color="#304263")
            ax.set_title("Inbound Batch Assortment mix")
            ax.set_ylabel("SKU Volume")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            df["primary_color"].value_counts().plot(kind="bar", ax=ax, color="#466291")
            ax.set_title("Primary Color Distribution")
            ax.set_ylabel("SKU Volume")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

        st.subheader("📦 Verified Data Payload Matrix")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Structured Catalog Data (CSV)",
            data=csv,
            file_name="fashion_catalog_export.csv",
            mime="text/csv",
            use_container_width=True
        )