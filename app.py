import streamlit as st
from groq import Groq
import base64
import json
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fashion Image Analyzer", page_icon="👗", layout="centered")

st.title("👗 Fashion Image Analyzer")
st.write(
    "Upload one or more clothing/product images. The model will analyze each one and "
    "extract structured tags: category, color, style, occasion, and a short description."
)

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    api_key = st.text_input("Enter your Groq API key", type="password")

if not api_key:
    st.info("Enter your Groq API key above (or set it in Streamlit secrets) to continue.")
    st.stop()

client = Groq(api_key=api_key)

uploaded_files = st.file_uploader(
    "Upload image(s)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
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
            st.image(uploaded_file, width=200)

        media_type = uploaded_file.type
        image_bytes = uploaded_file.getvalue()

        with col2:
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                try:
                    result = analyze_image(image_bytes, media_type)
                    result["filename"] = uploaded_file.name
                    results.append(result)

                    st.markdown(f"**Category:** {result.get('category')}")
                    st.markdown(f"**Primary color:** {result.get('primary_color')}")
                    st.markdown(f"**Secondary colors:** {', '.join(result.get('secondary_colors', []))}")
                    st.markdown(f"**Style:** {result.get('style')}")
                    st.markdown(f"**Occasion:** {result.get('occasion')}")
                    st.markdown(f"**Target demographic:** {result.get('target_demographic')}")
                    st.markdown(f"**Fabric guess:** {result.get('fabric_guess')}")
                    st.markdown(f"**Description:** {result.get('description')}")

                except json.JSONDecodeError:
                    st.error("Could not parse model response as JSON. Try again.")
                except Exception as e:
                    st.error(f"Error: {e}")

    if len(results) > 1:
        st.divider()
        st.subheader("📊 Batch Summary")

        df = pd.DataFrame(results)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Category distribution**")
            fig, ax = plt.subplots()
            df["category"].value_counts().plot(kind="bar", ax=ax)
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with col2:
            st.write("**Primary color distribution**")
            fig, ax = plt.subplots()
            df["primary_color"].value_counts().plot(kind="bar", ax=ax)
            ax.set_ylabel("Count")
            st.pyplot(fig)

        st.write("**Full results table**")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results as CSV", csv, "fashion_analysis.csv", "text/csv")