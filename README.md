# 🧠 MediSense: A Healthcare Claims Credibility Checker

*Repo Authors: Ananya Nimbalkar & Sai Nandini Peesapati*

MediSense is an AI-powered tool designed to **evaluate the credibility of healthcare claims** and provide **reasoned explanations** using transformer-based models. Tailored for **young adults transitioning from pediatric to adult care**, MediSense equips users with the confidence to distinguish between credible medical facts and misinformation.

---

## 📌 Problem Statement & Overview

### 🎯 Clarity & Relevance
Healthcare misinformation is rampant online, especially on social media. For young adults navigating their own care for the first time, **distinguishing truth from misinformation** can be daunting. MediSense addresses this issue by using **Generative AI (GenAI)** to assess medical claims and explain its reasoning.

### 👥 Context & Impact
- **Audience**: Young adults (e.g., 17–25) transitioning from pediatric to general care.
- **Impact**: Encourages informed decision-making, builds digital health literacy, and supports trust in evidence-based medicine.

### 🔍 Scope & Feasibility
This project was built over four weeks, leveraging two key transformers with no need for massive retraining or compute infrastructure.

### 🧠 Technical Fit
Unlike simple prompting, MediSense:
- **Fine-tunes BioBERT** for classification.
- Uses **LaMini-Flan-T5** for explanation generation.
- Employs **synthetic data augmentation** to mitigate bias and improve performance.

---

## ⚙️ Methodology & Complexity

### ✅ Fine-Tuning BioBERT
We fine-tuned [BioBERT](https://arxiv.org/abs/1901.08746) (a domain-specific BERT for biomedical text) using a labeled dataset of healthcare claims:
- **Task**: Binary classification (credible vs. not credible).
- **Loss Function**: Cross-entropy.
- **Output**: Softmax confidence score → interpreted as a credibility percentage.

> 📁 See [`train_biobert.ipynb`](./train_biobert.ipynb)

---

### ✅ Synthetic Data Generation
We used generative prompting with LaMini-Flan-T5 to create **100+ credible claims** and corresponding **explanations**. This helped correct bias where most real-world claims were classified as **false**.

- **Prompting Strategy**: Instructional, emphasizing high-credibility scenarios.
- **Post-processing**: Cleaned outputs, filtered claims, and assigned synthetic scores between 71–95%.
- **Output**: `synthetic_claim_explanations.csv` used by the app.

> 📁 See [`generate_synthetic_claims.ipynb`](./generate_synthetic_claims.ipynb)

---

## 🧬 Model Architecture: How MediSense Works

MediSense uses a **dual-transformer pipeline** built on top of Hugging Face Transformers. Below is the architecture of each component:

---

### 🧪 1. BioBERT Classifier (Encoder-based)

We fine-tuned **BioBERT** (based on BERT architecture) for binary classification of healthcare claims.

#### 📐 Key Components:
- **Input**: Text claim → tokenized using WordPiece
- **Embedding Layer**: Positional + token embeddings
- **Transformer Encoder Layers**: 12 layers of multi-head self-attention + feed-forward
- **[CLS] Token Output**: Passed through classification head
- **Output**: 2 logits → Softmax → `score ∈ [0,100]%`

#### 🔁 Pseudocode:

```python
input_ids = tokenizer(claim, return_tensors="pt")
outputs = model(**input_ids)
logits = outputs.logits
probs = softmax(logits)
credibility = probs[1] * 100
```

### 💬 2. LaMini-Flan-T5 (Encoder-Decoder for Explanation)

We used **LaMini-Flan-T5**, a lightweight instruction-tuned variant of T5, to generate concise natural language **explanations** for why a healthcare claim is likely accurate — conditioned on the **claim text** and its associated **credibility score**.

#### 📐 Key Components
- **🧾 Input Format**:
    Claim: {claim}
    Credibility: {score}%
    Explain why this claim is likely accurate.
- **Encoder**: Uses multi-head self-attention to encode the prompt into latent embeddings.
- **Decoder**: Autoregressively generates the explanation, one token at a time.
- **Output**: A short 1–3 sentence **justification** for why the claim may be considered credible.

This architecture allows the model to condition explanation generation not only on the claim content but also the numeric confidence (score) from the classifier.

#### 🔁 Pseudocode:

```python
prompt = f"Claim: {claim}\nCredibility: {score}%\nExplain why this claim is likely accurate."
output = reasoning_pipeline(prompt)
```

#### 🔀 Combined Flow
          ┌────────────────────┐
          │  User or Synthetic │
          │     Input Claim    │
          └─────────┬──────────┘
                    ▼
         ┌───────────────────────┐
         │   BioBERT Classifier  │ 🔍
         └─────────┬─────────────┘
                   ▼
      Credibility Score (0–100)%
                   │
                   ▼
    ┌────────────────────────────────┐
    │     LaMini-Flan-T5 Reasoner    │ 🧠
    └────────────────────────────────┘
                   │
                   ▼
         Explanation Generated 💬

---

## 💻 Implementation & Demo

### 🧪 Streamlit App
The full system is operational in [`app.py`](./app.py). Users can:
- Input a custom healthcare claim.
- OR select an AI-generated synthetic claim.
- Get a **credibility score** and a **reasoned explanation**.

```bash
python -m streamlit run app.py
```
## 🗂️ Input / Output Flow

| Step                   | Input                            | Output                            |
|------------------------|----------------------------------|-----------------------------------|
| BioBERT Fine-Tuning    | Labeled claims CSV               | `biobert_misinformation_model/`   |
| Reasoning (Flan-T5)    | Prompted claims + scores         | Explanations (inline)             |
| Synthetic Generation   | Seed topics + Flan-T5            | `synthetic_claim_explanations.csv` |
| Streamlit Interface    | Claim from user or synthetic     | Score + Explanation               |

---

## 📏 Assessment & Evaluation

### 📊 Metrics

We evaluated generated explanations using:

- **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation)
- **BLEU** (Bilingual Evaluation Understudy)

📁 See [`lamini_reasoning_eval.ipynb`](./lamini_reasoning_eval.ipynb)

---

### 📉 Limitations

- No factual grounding in explanations — depends on pretrained model bias.
- BioBERT still slightly skews toward conservative (false) predictions.
- Score is not calibrated to clinical certainty.

---

## 🧾 Model & Data Cards

| Component         | Model                    | Notes                                       |
|------------------|--------------------------|---------------------------------------------|
| Classifier       | Fine-tuned BioBERT       | Binary classification                       |
| Reasoning Model  | LaMini-Flan-T5-248M      | Instruction-tuned LLM                       |
| Synthetic Claims | Generated via Flan-T5    | Tagged as AI-generated for transparency     |
| License          | Apache 2.0               | Fair-use for academic and research purposes |

> ⚠️ **Bias Note**: AI-generated claims may reinforce health norms and lack diversity. Use with caution in real-world clinical decision-making.

---

## 🧠 Critical Analysis

### What did we learn?

- Combining domain-specific models with instruction-tuned LLMs provides both **accuracy** and **interpretability**.
- Synthetic data can dramatically **rebalance a skewed classification task** and improve the model's generalization.

### Next steps?

- Add **retrieval-based fact-checking (RAG)**.
- Incorporate **factual grounding** using resources like **PubMed** or **MedQA**.
- Expand to other audiences: **parents, caregivers**, and **low-literacy populations**.

---

## 📚 Documentation & Resource Links

### 🧭 Repo Guide

```bash
├── app.py                           # Streamlit UI
├── train_biobert.ipynb              # BioBERT training
├── generate_synthetic_claims.ipynb  # Claim generation
├── lamini_reasoning_eval.ipynb      # Evaluation of explanations
├── synthetic_claim_explanations.csv # Output dataset
```
### 🔗 External Resources

- 📄 [BioBERT Paper](https://arxiv.org/abs/1901.08746)
- 📄 [LaMini-Flan Paper](https://arxiv.org/pdf/2304.14402)
- 🧰 [Transformers Library](https://huggingface.co/docs/transformers/en/index)
- 📹 [Named Entity Recognition with BioBERT](https://youtu.be/zjYs52Met8E?si=9geahLbvp7QAHIMH)
- 


