# Sentiment Analysis: RNN vs LSTM vs GRU

## Project Overview

This project focuses on **binary sentiment classification** of text data using both **traditional machine learning** and **deep learning sequence models**.

The main objective is to compare the performance of:

- TF-IDF + Logistic Regression (Baseline)
- Simple RNN
- LSTM
- GRU

The goal is to understand how different modeling approaches perform on sentiment analysis and whether deep learning models outperform classical methods.

---

## Dataset

- Dataset: **IMDB Movie Reviews**
- Task: Binary classification (Positive / Negative)
- Target:
  - `positive → 1`
  - `negative → 0`

---

## Data Preprocessing

Text preprocessing was performed using **spaCy** and regex-based cleaning:

- Lowercasing
- HTML tag removal
- URL removal
- Non-alphabetic character filtering
- Stopword removal (with **negation words preserved**)
- Lemmatization

Example:

- "I do not like this movie at all"
- → "not like movie"

---

## Feature Engineering

Two different approaches were used:

### Traditional ML
- TF-IDF Vectorization
- Unigram + Bigram features
- Max features: 10,000

### Deep Learning
- Tokenization
- Sequence encoding
- Padding (fixed length sequences)
- Trainable embedding layer

---

## Models

### 1. Baseline Model
- TF-IDF + Logistic Regression

### 2. Deep Learning Models
- Simple RNN
- LSTM
- GRU

All deep learning models share:
- Same dataset splits
- Same preprocessing
- Same embedding dimension
- Same training setup

---

## Results

### Performance Comparison

| Model | Val Acc | Test Acc | Val F1 | Test F1 |
|------|--------|---------|--------|--------|
| TF-IDF + Logistic Regression | 0.891 | **0.897** | 0.893 | **0.898** |
| GRU | 0.876 | 0.887 | 0.876 | 0.887 |
| LSTM | 0.865 | 0.872 | 0.867 | 0.873 |
| Simple RNN | 0.808 | 0.812 | 0.799 | 0.805 |

---

## Best Model

**TF-IDF + Logistic Regression achieved the best performance**

- Test Accuracy: **0.8968**
- Test F1 Score: **0.8981**

---

##  Key Insights

### 1. Classical ML is very strong
Despite the use of deep learning models, the **TF-IDF + Logistic Regression baseline outperformed all sequence models**.

This indicates that:
- The dataset is well-structured
- Bag-of-words representations are highly effective for this task

---

### 2. GRU performed best among deep learning models
- GRU outperformed both LSTM and Simple RNN
- Shows strong balance between performance and model complexity

---

### 3. Simple RNN limitations
- Lowest performance among all models
- Likely affected by **vanishing gradient problem**
- Struggles with long-term dependencies

---

### 4. LSTM vs GRU
- LSTM performed well but slightly below GRU
- GRU achieved comparable or better performance with fewer parameters

---

### 5. Deep Learning is not always better
This project demonstrates an important lesson:

> **More complex models do not always outperform simpler ones.**

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Future Improvements

- Pretrained embeddings (GloVe, Word2Vec)
- Bidirectional LSTM / GRU
- Attention mechanisms
- Transformer-based models (BERT)
- Hyperparameter tuning

---
