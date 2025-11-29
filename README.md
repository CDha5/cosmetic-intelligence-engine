# Cosmetic Intelligence Engine 🧪

A research-based application comparing TF-IDF and Binary models for cosmetic ingredient analysis.

## 📝 Project Overview
This project addresses the limitations of binary bag-of-words models in cosmetic recommendation systems. It implements a **TF-IDF (Term Frequency-Inverse Document Frequency)** model to better capture the importance of functional ingredients compared to common fillers.

The system serves two purposes:
1. **A Practical Tool:** Helping consumers find personalized product alternatives and assess risk.
2. **A Research Study:** Providing empirical evidence that weighted vector models outperform binary models in this domain.

## 🚀 Features
### 1. Personalized Recommender Tool
* **Skin-Type Adaptation:** Filters products based on Dry, Oily, Combination, or Normal skin.
* **Smart Similarity:** Uses TF-IDF and Cosine Similarity to find functionally similar products.
* **Advanced Filters:** Filter by Price, Rank, and "Dupe Finder" mode.
* **Functional Analysis:** Explains *why* a product is recommended by breaking down shared ingredients and their functions.

### 2. Sensitivity Predictor
* **Machine Learning Model:** A Logistic Regression model trained on 1,400+ products.
* **Risk Assessment:** Predicts the probability of a product being suitable for sensitive skin.

### 3. Research Findings Interface
* **Quantitative Proof:** Displays the results of the comparative experiment against a "ground truth" dataset.
* **Qualitative Case Study:** Showcases a side-by-side comparison of recommendation relevance.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-learn (TF-IDF, Logistic Regression, Cosine Similarity)
* **Data Manipulation:** Pandas, NumPy

## 📦 How to Run Locally
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/CDha5/cosmetic-intelligence-engine.git](https://github.com/CDha5/cosmetic-intelligence-engine.git)