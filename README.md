# Phish or Legit?

A small Streamlit app that classifies a pasted message as **phishing-style** or **legit** using a TF-IDF + logistic regression model trained on a synthetic dataset of fictional-brand messages.

![App screenshot](img.png)

## Purpose

- Demonstrate a text-classification pipeline (TF-IDF features + logistic regression) on a synthetic phishing/legit message dataset.
- Show which words push the prediction toward "phish" (red) or "legit" (green).
- Provide calibrated vs. uncalibrated score comparison.

> Note: the dataset is synthetic and labels are "phishing-style" vs "legit", not real-world ground truth.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL printed in the terminal (usually http://localhost:8501).

## Project structure

- `app.py` — Streamlit UI
- `src/pipeline.py` — data loading, model building, prediction, explanations
- `data/messages.csv` — synthetic dataset
- `data/make_data.py` — dataset generator
"""