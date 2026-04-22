# Personality-Trait-Prediction-from-Text-Using-Machine-Learning
Predicts Introvert/Extrovert and Thinking/Feeling from text using Logistic Regression. Compares Naive Bayes, uses TF‑IDF with bigrams, and includes word clouds + live demo widget.

## Overview

This project applies machine learning to predict personality traits from written text. Using a dataset of forum posts labeled with Myers-Briggs (MBTI) types, we build classifiers for two observable dimensions:

- **Introvert vs Extravert** (social style)
- **Thinking vs Feeling** (decision style)

We compare **Naive Bayes** (baseline) with **Logistic Regression** (with class balancing). The best model achieves ~85% accuracy. The repository includes data preprocessing, TF‑IDF vectorization (with bigrams), word clouds for interpretability, and an interactive Jupyter widget for live testing.

## Dataset

- **Source:** [MBTI Dataset on Kaggle](https://www.kaggle.com/datasnaek/mbti-type)
- **Size:** 8,675 forum posts
- **Each row:** MBTI type + user post text
- **Preprocessing:**  
  - Remove `||||` separators  
  - Remove MBTI type names (e.g., "INTJ", "ENTP") and cognitive functions (e.g., "Ni", "Te")  


### Feature Extraction
- **TF‑IDF** with bigrams (`ngram_range=(1,2)`)
- **max_features** = 5,000
- Stop words removed

### Models
1. **Naive Bayes (MultinomialNB)** – fast, simple baseline
2. **Logistic Regression** – with `class_weight='balanced'` to handle imbalanced classes (more Introverts than Extraverts)

### Evaluation
- Train/test split: 80/20

## Results

| Model | I vs E Accuracy | T vs F Accuracy |
|-------|----------------|-----------------|
| Naive Bayes | 78.5% | 80.1% |
| Logistic Regression (balanced) | **84.6%** | **85.3%** |

**Key improvements from class balancing:**  
- Recall for Extravert improved from 3% (Naive Bayes) to 73% (Logistic Regression)

### Word Clouds (interpretability)
- **Introvert‑associated:** quiet, family, feel, mind, dream, hate  
- **Extravert‑associated:** fun, friends, guys, awesome, debate, love

These patterns align with psychological expectations, validating the model.

## Live Demo

An interactive Jupyter widget allows you to type any sentence and get a personality prediction in real time. Run the notebook and execute the cell containing the widget code.

