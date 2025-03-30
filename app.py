# app.py
# Run with: python -m streamlit run app.py

import torch
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Healthcare Claims Credibility Checker")

# -----------------------------
# 🚂 Load BioBERT Classifier
# -----------------------------
biobert_path = Path("biobert_misinformation_model")

try:
    biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_path)
    biobert_model = AutoModelForSequenceClassification.from_pretrained(
        biobert_path,
        local_files_only=True,
        trust_remote_code=True
    )
    st.sidebar.success("✅ BioBERT model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load BioBERT model: {e}")
    st.stop()

# -----------------------------
# 🧠 Load Reasoning Model
# -----------------------------
try:
    reasoning_tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
    reasoning_model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
    reasoning_pipeline = pipeline("text2text-generation", model=reasoning_model, tokenizer=reasoning_tokenizer, device=-1)
    st.sidebar.success("✅ LaMini-Flan reasoning model (248M) loaded.")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load reasoning model: {e}")
    st.stop()

# -----------------------------
# 📁 Load Synthetic Claims from CSV
# -----------------------------
try:
    synthetic_df = pd.read_csv("synthetic_claim_explanations.csv")
    synthetic_df = synthetic_df.dropna(subset=["claim"])  # in case any rows are incomplete
    st.sidebar.success(f"✅ Loaded {len(synthetic_df)} synthetic claims.")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load synthetic claims: {e}")
    synthetic_df = pd.DataFrame(columns=["claim", "score", "explanation"])

# -----------------------------
# 🔍 Classify Claim
# -----------------------------
def classify_claim(text):
    inputs = biobert_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = biobert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return round(probs[0][1].item() * 100, 2)

# -----------------------------
# 💡 Generate Explanation
# -----------------------------
def generate_reasoning(claim, score):
    reasoning_type = "accurate" if score > 70 else "misinformation"
    prompt = f"Claim: {claim}\nCredibility: {int(score)}%\nExplain why this claim is likely {reasoning_type}."
    return reasoning_pipeline(prompt, max_length=128, num_return_sequences=1)[0]["generated_text"].strip()

# -----------------------------
# 🎯 Streamlit UI
# -----------------------------
st.title("🧠 MediSense: A Healthcare Claims Credibility Checker")
st.markdown("Check whether a healthcare claim is **credible or not** and get a short explanation using AI.")

# Checkbox for synthetic mode
use_synthetic = st.checkbox("📋 Try a random synthetic claim")

# Initialize session state to hold selected synthetic
if "synthetic_index" not in st.session_state:
    st.session_state.synthetic_index = None

# Pick random synthetic row only once
if use_synthetic and st.session_state.synthetic_index is None and not synthetic_df.empty:
    st.session_state.synthetic_index = random.randint(0, len(synthetic_df) - 1)

# Handle synthetic mode
if use_synthetic and st.session_state.synthetic_index is not None:
    row = synthetic_df.iloc[st.session_state.synthetic_index]
    claim_input = row["claim"]
    preset_score = row["score"]
    preset_explanation = row["explanation"]

    st.markdown("🧪 **Synthetic Claim (AI-generated):**")
    st.info(claim_input)
else:
    st.session_state.synthetic_index = None
    claim_input = st.text_area("Enter a healthcare claim below:", height=120)

# Run prediction
if st.button("🔎 Check Credibility"):
    if not claim_input.strip():
        st.warning("Please enter a healthcare claim.")
    else:
        with st.spinner("Analyzing..."):
            if use_synthetic and st.session_state.synthetic_index is not None:
                score = preset_score
                explanation = preset_explanation
            else:
                score = classify_claim(claim_input)
                explanation = generate_reasoning(claim_input, score)

        st.markdown(f"### 📊 Credibility Score: `{score:.2f}%`")
        st.markdown("### 💬 Explanation:")
        st.write(explanation)

# Footer
st.markdown("---")
st.caption("Built with 💙 using BioBERT + LaMini-Flan-T5")
