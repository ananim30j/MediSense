# 🧠 MediSense: A Healthcare Claims Credibility Checker

*Repo Authors: Ananya Nimbalkar & Sai Nandini Peesapati*

MediSense is an AI-powered tool designed to **evaluate the credibility of healthcare claims** and provide **reasoned explanations** using transformer-based models. Tailored for **young adults transitioning from pediatric to adult care**, MediSense equips users with the confidence to distinguish between credible medical facts and misinformation.

---

## 📌 Problem Statement & Overview

### 🎯 Clarity & Relevance
Healthcare misinformation is rampant online, especially on social media. For young adults navigating their own care for the first time, **distinguishing truth from misinformation** can be daunting. MediSense addresses this issue by using **Generative AI (GenAI)** to assess medical claims and explain its reasoning.

### 👥 Context & Impact
- **Audience**: Young adults (e.g., 17–25) transitioning from pediatric to general care.
- **Challenge**: As these individuals begin managing their own healthcare, they are particularly vulnerable to misinformation online — especially through social media, peer groups, and unverified content.
- **Solution**: MediSense empowers this demographic with a tool that not only flags questionable claims but also **explains the reasoning** in simple, understandable terms. This fosters health literacy, supports independent decision-making, and encourages trust in evidence-based medicine.
- **Long-Term Impact**: By equipping young adults with the ability to critically evaluate medical information early in life, MediSense helps build lasting habits of healthy skepticism and informed consent.

### 🔍 Scope & Feasibility
MediSense is a complete, end-to-end prototype that demonstrates the viability of using **fine-tuned biomedical transformers** and **instruction-tuned generative models** for misinformation detection — all without requiring high-end compute or access to proprietary APIs.

### 🧠 Technical Fit
Unlike simple prompting, MediSense:
- **Fine-tunes BioBERT** for classification.
- Uses **LaMini-Flan-T5** for explanation generation.
- Employs **synthetic data augmentation** to mitigate bias and improve performance.

---

## ⚙️ Methodology & Complexity

### ✅ Fine-Tuning BioBERT (Model Training)
We fine-tuned [BioBERT](https://arxiv.org/abs/1901.08746), a biomedical variant of BERT, on a curated dataset of real healthcare claims labeled as *credible* or *not credible*.

#### 📁 Dataset Preparation
- Manually cleaned claims dataset with binary labels (0 = misinformation, 1 = credible).
- Tokenized using WordPiece tokenizer specific to BioBERT.
- Split into training/validation sets (80/20).
- Class imbalance noted: ~75% labeled false → key motivation for augmentation.

#### 🛠️ Training Details
- **Batch size**: 16  
- **Epochs**: 4  
- **Optimizer**: AdamW  
- **Learning rate**: 2e-5  
- **Loss function**: CrossEntropyLoss

#### 📈 Why Fine-Tune?
Off-the-shelf models lacked domain specificity and treated medical claims as generic text classification. BioBERT, pre-trained on PubMed and PMC abstracts, gave us a **domain-aware foundation**, and fine-tuning helped specialize it for **credibility detection**, not just medical NER or QA.

#### ✅ Outcome
- Model outputs a confidence score (0–100%) interpretable by end-users.
- Added explainability layer through downstream reasoning pipeline.
- Observed improvement in consistency over baseline models like vanilla BERT.

> 📁 See [`train_biobert.ipynb`](./train_biobert.ipynb) for full training logs and code.

---

### ✅ Synthetic Data Generation (Bias Mitigation)

Due to **high false claim prevalence** in the original dataset, we generated **high-quality synthetic data** using [LaMini-Flan-T5](https://arxiv.org/abs/2304.14402). These synthetic samples helped rebalance and enrich the credibility space.

#### 🧠 Generation Approach
- Seeded the model with instructional prompts encouraging **credible health topics**.
- Generated 100+ **plausible and fact-aligned** claims.
- Assigned simulated credibility scores (71–95%) to each.
- Generated explanations for each using the same reasoning model.

#### 🧹 Post-Processing
- Deduplicated and cleaned the claims.
- Removed phrases like "True" or "This is a true claim" from outputs.
- Final output stored in `synthetic_claim_explanations.csv`.

#### 📊 Impact of Synthetic Data
- **Addressed class imbalance**: now users see credible claims alongside false ones.
- **Expanded domain coverage**: included lesser-represented topics like mental health, hydration, sleep science, etc.
- **Reduced hallucination** in explanations due to more consistent training input structure.

#### ⚠️ Pitfalls & Mitigation
- Some outputs were generic (e.g., “Agreed”) — filtered during quality check.
- No hard grounding → made sure outputs remained plausible by seeding with realistic prompts and validating with BioBERT predictions.

> 📁 See [`generate_synthetic_claims.ipynb`](./generate_synthetic_claims.ipynb)

---

## 🧬 Model Architecture: How MediSense Works

MediSense uses a **dual-transformer pipeline** built on top of Hugging Face Transformers. Below is the architecture of each component:

---

## 🧬 Model Architecture: How MediSense Works

MediSense uses a **two-part transformer pipeline**, where both components are derived from the **Transformer architecture** introduced by Vaswani et al. This pipeline combines the **encoder-based power of BERT** with the **encoder-decoder reasoning capabilities of T5**.

---

### 🧪 1. BioBERT Classifier (Transformer Encoder)

**BioBERT** is built on top of the original **BERT-base** transformer architecture, consisting of **12 layers** of self-attention, trained specifically on biomedical text.

#### 🧠 Transformer Stack
- **Input**: Raw text claim
- **Tokenizer**: WordPiece tokenizer → token IDs
- **Embeddings**: Token, position, and segment embeddings are added together
- **Encoder**:
  - **Multi-head Self-Attention**: Each token attends to every other token
  - **Feedforward Layers**: Non-linear transformations applied after attention
  - **Residual Connections + LayerNorm**: Maintain stability and speed up convergence
- **[CLS] Token**: Output from the first position is passed to a classification head
- **Classification Head**: Fully connected layer + Softmax

#### 🔁 Pseudocode
```python
# Tokenization
tokens = biobert_tokenizer(claim_text, return_tensors="pt", truncation=True)

# Forward pass through transformer encoder
with torch.no_grad():
    outputs = biobert_model(**tokens)   # includes hidden states

# Logits from classification head (2 outputs for binary classes)
logits = outputs.logits

# Softmax for probability distribution over labels
probabilities = torch.nn.functional.softmax(logits, dim=1)

# Take credibility score (label 1 = credible)
credibility_score = probabilities[0][1].item() * 100
```

### 💬 2. LaMini-Flan-T5 (Transformer Encoder-Decoder)

**T5 (Text-To-Text Transfer Transformer)** is a **fully transformer-based model** that uses:

- **Encoder**: To understand the task and context (claim + credibility)
- **Decoder**: To generate step-by-step reasoning as a natural language explanation

We used the **LaMini-Flan-T5-248M**, a compact instruction-tuned variant of T5 optimized for constrained environments.

---

#### 🧠 Transformer Flow

- **Input Format**:
  ```
  Claim: {claim}
  Credibility: {score}%
  Explain why this claim is likely accurate.
  ```

- **Encoder**:
  - Processes the prompt using self-attention across tokens
  - Builds contextual embeddings

- **Decoder**:
  - Uses encoder’s final layer as input
  - Applies masked self-attention to generate one token at a time
  - Uses teacher forcing during training, greedy decoding during inference

---

#### 🔁 Pseudocode
```python
# Format input with credibility score
prompt = f"Claim: {claim_text}\nCredibility: {score:.0f}%\nExplain why this claim is likely accurate."

# Generate explanation using encoder-decoder attention
explanation = reasoning_pipeline(
    prompt,
    max_length=128,
    do_sample=False,
    num_return_sequences=1
)[0]["generated_text"].strip()
```

---

#### 🔀 Combined Transformer Flow

```
          ┌────────────────────────────┐
          │  User or Synthetic Claim   │
          └────────────┬───────────────┘
                       ▼
         ┌────────────────────────────┐
         │    BioBERT Transformer     │ 🔍
         │ (Encoder-Only Architecture)│
         └────────────┬───────────────┘
                       ▼
         Credibility Score (0–100%) 🔢
                       │
                       ▼
    ┌─────────────────────────────────────┐
    │  LaMini-Flan-T5 Transformer         │ 🧠
    │ (Encoder-Decoder Architecture)      │
    └─────────────────────────────────────┘
                       │
                       ▼
         Natural Language Explanation 💬
```

> ⚙️ Both components are based on the **attention mechanism**, enabling them to contextualize information and support interpretability.

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

### 🧾 Intermediate Outputs & Debugging Logs

Throughout development, each notebook generated artifacts that enabled step-by-step verification and experimentation:

| Notebook                           | Intermediate Outputs                                               | Purpose                                                       |
|------------------------------------|--------------------------------------------------------------------|---------------------------------------------------------------|
| `train_biobert.ipynb`              | - Training loss per epoch<br>- Accuracy trends<br>- Checkpoint folder `biobert_misinformation_model/` | Validate model convergence, inspect overfitting/underfitting |
| `lamini_reasoning_eval.ipynb`     | - Generated explanations<br>- CSV with predictions<br>- BLEU & ROUGE score summaries | Evaluate explanation quality using text similarity metrics   |
| `generate_synthetic_claims.ipynb` | - Synthetic claim list<br>- Score assignment logs<br>- `synthetic_claim_explanations.csv` | Verify prompt effectiveness, score distributions             |

> 🔍 Debugging prints were added in each notebook to help examine prediction quality, prompt output quality, and detect anomalies (e.g., hallucinations, duplicates, or low-variance generations).

---

## 📏 Assessment & Evaluation

### 📊 Metrics

To evaluate MediSense, we used both **classification** and **generation** metrics appropriate to each model component:

---

#### 🧪 BioBERT Classification Evaluation

- **Accuracy**: `82.49%` – indicates the percentage of claims correctly classified.
- **Precision**: `24.61%` – of the claims predicted as credible, how many were truly credible.
- **Recall**: `32.00%` – of the truly credible claims, how many were retrieved.
- **F1 Score**: `27.82%` – harmonic mean of precision and recall.
- **Loss**: `0.4437` – cross-entropy loss during evaluation.

These scores reflect a moderately imbalanced dataset and highlight the challenge of classifying claims as credible when most real-world data skews toward misinformation. This justified the need for synthetic data augmentation.

---

#### 💬 LaMini-Flan-T5 Explanation Evaluation

We used two text generation metrics to assess the quality of explanations:

- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**:
  - `ROUGE-1`: `0.0815`
  - `ROUGE-2`: `0.0192`
  - `ROUGE-L`: `0.0610`
  - `ROUGE-Lsum`: `0.0608`

  These capture overlap between generated and reference explanations (based on unigrams, bigrams, and longest common subsequences). While relatively low, this is expected for open-ended generation where word choice varies widely.

- **BLEU (Bilingual Evaluation Understudy)**: `0.47`

  BLEU measures n-gram precision against reference text and is useful for short, structured responses. A BLEU score of 0.47 shows moderate alignment with human-written ground truth.

---

### 📉 Limitations

Despite promising results, MediSense has several limitations:

- **Factual Grounding**: The reasoning model (LaMini-Flan-T5) is not grounded in external knowledge bases like PubMed or MedQA. Explanations are purely generated from pretraining and may lack verifiable citations.

- **Model Bias**: BioBERT and LaMini-Flan were pretrained on domain-specific and general corpora, respectively. This introduces potential bias, especially in interpreting controversial or nuanced health claims.

- **Data Coverage**: While MediSense performs well on general wellness and common medical topics, the underlying dataset (real and synthetic) lacks comprehensive coverage of the entire healthcare spectrum. Specialized fields like rare diseases, surgical techniques, or pharmacogenomics are underrepresented.

- **Computational Constraints**: Due to limited compute availability, we could not retrain or fine-tune larger generative models. In fact, full training runs were estimated at over **55 hours** on a GPU. We instead opted for zero-shot or instruction-tuned models (e.g., LaMini-Flan-T5-248M), which trade depth for accessibility.

- **Score Calibration**: The credibility score, derived from softmax probabilities, is not calibrated to real-world clinical certainty. Users should not interpret a high score as medical advice.

- **Explanation Variability**: Generated explanations may repeat, be vague, or hallucinate supporting facts—especially when prompts are ambiguous or highly subjective.

---

## 🧾 Model & Data Cards

| Component         | Model                    | Notes                                                                 |
|------------------|--------------------------|-----------------------------------------------------------------------|
| Classifier       | Fine-tuned BioBERT       | Binary classification (credible vs. not credible); encoder-only model |
| Reasoning Model  | LaMini-Flan-T5-248M      | Instruction-tuned encoder-decoder for NLG-based explanation           |
| Synthetic Claims | Generated via Flan-T5    | Used only for augmentation; clearly tagged as AI-generated            |
| License          | Apache 2.0               | Open-source, educational/research usage                               |

### ⚙️ Model Architectures

- **BioBERT**: Based on the BERT-Base (12-layer Transformer encoder), pretrained on PubMed abstracts and PMC full-texts. We fine-tuned it for binary classification using a cross-entropy loss head.
- **LaMini-Flan-T5-248M**: A 248M parameter version of the Flan-T5 model (encoder-decoder), instruction-tuned on diverse prompts. Used for zero-shot explanation generation with minimal resource cost.

### ✅ Intended Use

- **Educational Tool**: MediSense is intended to help young adults develop critical thinking when encountering medical claims.
- **Health Literacy Support**: Not intended for medical diagnosis or clinical decisions, but as a supplementary tool to foster digital health awareness.
- **Academic Showcase**: Designed as a proof-of-concept project for GenAI education in a Transformers course.

### ⚠️ Ethical & Bias Considerations

- **Bias from Pretraining Corpora**: BioBERT is trained on biomedical literature, which may skew conservative (toward medically vetted sources), and LaMini-Flan-T5 on general internet data, which may reflect public discourse biases.

- **Synthetic Data Reinforcement**: AI-generated claims could reinforce dominant narratives (e.g., Western-centric health practices, overemphasis on fitness/diet over social determinants of health). We mitigate this by:
  - Tagging synthetic claims in the app.
  - Balancing between real and synthetic datasets.
  - Encouraging transparency via open-source access to data and code.

- **Misinformation Risk**: While the app discourages false claims, generated explanations are not fact-checked against real-time medical databases and should not be used in clinical settings.

> 🛑 Always consult a licensed healthcare provider before acting on any health-related advice.

---

## 🧠 Critical Analysis

### 💡 What did we learn?

MediSense revealed that pairing **domain-specific transformers** like BioBERT with **instruction-tuned LLMs** like LaMini-Flan-T5 can yield highly interpretable, real-time assessments of healthcare information. We observed that:

- **Fine-tuning BioBERT** allowed the model to learn nuanced patterns in medical text, outperforming generic LLM classifiers.
- **Synthetic data** played a pivotal role in correcting bias and expanding the scope of positive claims, ensuring the model didn't default to "false" due to data imbalance.
- Using **transformers in a modular pipeline** allowed for separation of logic — classification vs. reasoning — improving both clarity and flexibility.

### 🔮 Next Steps

- **Integrate RAG (Retrieval-Augmented Generation)** to enhance factual grounding using PubMed or MedQA.
- Add **confidence calibration techniques** to interpret probability scores more clinically.
- Expand dataset to cover underrepresented health topics (e.g., mental health, reproductive care).
- Explore **few-shot instruction tuning** on LaMini-Flan to make explanations more factual and diverse.

---

## 📚 Documentation & Resource Links

### 🧭 Repo Guide

```bash
├── README.md                         # Project overview and instructions
├── app.py                            # Streamlit application for user interface
├── claim_explanations.csv            # Real claims and their explanations
├── claims.csv                        # Raw input claims for BioBERT fine-tuning
├── claims_with_predictions.csv       # Model outputs from BioBERT classification
├── generate_synthetic_claims.ipynb   # Generates synthetic healthcare claims
├── lamini_generated_explanations.csv # Raw LaMini-Flan-T5 generated explanations
├── lamini_reasoning_eval.ipynb       # Evaluation notebook for BLEU/ROUGE scores
├── synthetic_claim_explanations.csv  # Cleaned synthetic claims with scores/explanations
├── train_biobert.ipynb               # Fine-tuning notebook for BioBERT classifier
```
### 🔗 External Resources

- 📄 [BioBERT Paper](https://arxiv.org/abs/1901.08746)
- 📄 [LaMini-Flan Paper](https://arxiv.org/pdf/2304.14402)
- 🧰 [Transformers Library](https://huggingface.co/docs/transformers/en/index)
- 📹 [Named Entity Recognition with BioBERT](https://youtu.be/zjYs52Met8E?si=9geahLbvp7QAHIMH)

---

## 🧑‍💻 User Guide: Reproducing MediSense

Follow the steps below to reproduce this project end-to-end on your local machine:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ananim30j/MediSense.git
cd MediSense
```

### 2️⃣ Set Up the Environment

Create and activate a Python environment with the required libraries.

```
conda create -n medisense python=3.10
conda activate medisense
pip install -r requirements.txt  # Or manually install: transformers, torch, streamlit, pandas, etc.
```

### 3️⃣ Generate or Load Data

If starting from scratch:

🧪 a. Generate Synthetic Claims

Run the notebook below to generate 100+ high-credibility synthetic claims and their explanations:

```
jupyter notebook generate_synthetic_claims.ipynb
```

- Output: `synthetic_claim_explanations.csv`
- ⚠️ Make sure this file is saved in the same directory as `app.py`.

### 4️⃣ Fine-Tune BioBERT

Run the notebook below to fine-tune BioBERT on the labeled claim dataset:

```
jupyter notebook train_biobert.ipynb
```

- Output: A folder named `biobert_misinformation_model` containing:
    - `pytorch_model.bin`
    - `config.json`
    - `tokenizer_config.json`, etc.
- ⚠️ Ensure this folder is saved in the same directory as `app.py`.

### 5️⃣ Evaluate LaMini-Flan-T5

```
jupyter notebook lamini_reasoning_eval.ipynb
```

- This notebook evaluates explanation quality using BLEU/ROUGE metrics.

### 6️⃣ Verify LaMini-Flan Model Access

Ensure internet access is available for Hugging Face model loading:

- `MBZUAI/LaMini-Flan-T5-248M` is used directly in the app.
- No fine-tuning needed, so no folder output required.

### 7️⃣ Launch the Streamlit App

You're now ready to use MediSense!

```
streamlit run app.py
```

- App Capabilities:
    - ✍️ Enter a custom healthcare claim.
    - 🧠 OR try a synthetic (AI-generated) claim from the dropdown.
    - 📊 Get a credibility score and natural language explanation.

### 📂 Final Folder Structure

Make sure your final project folder looks like this:

```
├── app.py
├── README.md
├── train_biobert.ipynb
├── generate_synthetic_claims.ipynb
├── lamini_reasoning_eval.ipynb
├── claims.csv
├── claims_with_predictions.csv
├── claim_explanations.csv
├── lamini_generated_explanations.csv
├── synthetic_claim_explanations.csv
├── biobert_misinformation_model/        # ✅ Fine-tuned BioBERT model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
```

Now you're ready to explore, modify, or extend MediSense as a real-world GenAI pipeline for misinformation detection! 🧬✨