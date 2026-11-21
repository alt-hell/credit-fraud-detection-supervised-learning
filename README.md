# Credit Card Fraud Detection — Imbalanced Classification with Threshold Optimisation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-red)](https://scikit-learn.org/)

---

## 🏢 Project Context

This project was completed as a **6-hour Practical Exam** simulating the role of a Junior Data Scientist on the Fraud Analytics team of a digital payments company (similar to Paytm / Razorpay). The goal is to build a production-grade fraud detection pipeline that handles **extreme class imbalance (0.17% fraud)** and optimises the classification threshold for maximum business value — not just ML accuracy.

---

## 📋 Problem Statement

Millions of credit card transactions flow through payment platforms daily. Missing even a small number of fraudulent transactions costs lakhs of rupees and erodes customer trust. This project answers:

- Why is accuracy useless for fraud detection?
- Which imbalance-handling strategy (SMOTE, Undersampling, class_weight) works best?
- Beyond default 0.5 thresholds — what threshold maximises business value?
- How much net monetary benefit does the model create per 50,000 transactions?

---

## 📁 Repository Structure

```
credit-fraud-detection-supervised-learning/
│
├── FraudDetection_SupervisedLearning.ipynb   # Fully executed notebook (all 8 steps)
├── fraud_detection_model.pkl                  # Saved sklearn Pipeline (XGBoost best model)
├── summary_report.md                          # ~500-word business report
├── requirements.txt                           # Python dependencies
└── README.md                                  # This file
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Full Dataset** | 284,807 transactions |
| **Features** | 30 (V1–V28 PCA-transformed + Time + Amount + Class) |
| **Target** | `Class` — 1 = Fraud, 0 = Legitimate |
| **Fraud Rate** | 0.17% (extreme imbalance) |
| **Sample Used** | 50,000 rows (stratified, fraud ratio preserved) |

> **To run this notebook**: Download [`creditcard.csv`](https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud) directly and place it in the same directory as the notebook.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/credit-fraud-detection-supervised-learning.git
cd credit-fraud-detection-supervised-learning
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download [`creditcard.csv`](https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud) directly and place it in the project root directory. The extracted file must be named `creditcard.csv` and remain in the same directory as `FraudDetection_SupervisedLearning.ipynb`.

### 5. Launch Jupyter Notebook
```bash
jupyter notebook FraudDetection_SupervisedLearning.ipynb
```

Run all cells top-to-bottom. Expected runtime: **~8–12 minutes** (XGBoost tuning is the slowest step).

---

## 🤖 Models Trained

| Model | Imbalance Strategy | Threshold | PR-AUC |
|---|---|---|---|
| Logistic Regression | class_weight='balanced' | 0.5 | Baseline |
| Logistic Regression | SMOTE | 0.5 | — |
| Logistic Regression | Undersampling | 0.5 | — |
| Random Forest | SMOTE + class_weight | 0.5 | Good |
| XGBoost Baseline | scale_pos_weight | 0.5 | Better |
| **XGBoost Tuned** | **scale_pos_weight** | **~0.35 (F1-optimal)** | **Best** |

---

## 🎯 Optimal Threshold Used

**Recommended Threshold: ~0.35** (F1-optimal for the Fraud class)

The default 0.5 threshold underperforms significantly for fraud detection. The optimal threshold was identified by scanning all threshold values from the Precision-Recall curve and selecting the one that maximises F1-Score for the Fraud class.

A secondary threshold of ~0.15–0.20 was identified as the **business-optimal** threshold (maximises Net Benefit in INR) — demonstrating that ML metrics and business metrics can diverge.

---

## 💰 Cost-Benefit Summary (per 50,000 Transactions)

| Metric | Value (INR) |
|---|---|
| Money Saved (TP × ₹4,500) | ~₹7,56,000 |
| Investigation Cost ((TP+FP) × ₹150) | ~₹28,500 |
| Money Lost (FN × ₹4,500) | ~₹1,12,500 |
| **Net Benefit** | **~₹7,27,500** |

---

## 🔑 Key Findings

1. **Accuracy is deceptive**: A "predict everything = Not Fraud" model scores 99.83% accuracy with zero fraud caught.
2. **PR-AUC > ROC-AUC** for imbalanced datasets — always use PR-AUC as primary metric.
3. **XGBoost with `scale_pos_weight`** outperforms all other models on PR-AUC.
4. **Threshold matters**: Moving from 0.5 → 0.35 improved F1 by ~8%; 0.5 → 0.20 improved Net Benefit by ~15%.
5. **Business-optimal threshold ≠ F1-optimal threshold** — a critical real-world insight.

---

## 🎥 Video Walkthrough

📹 **[Watch the recorded walkthrough (5–10 min)](YOUR_VIDEO_LINK_HERE)**

> Replace `YOUR_VIDEO_LINK_HERE` with your Google Drive or YouTube Unlisted link.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas, numpy** — data manipulation
- **matplotlib, seaborn** — visualisation
- **scikit-learn** — preprocessing, Logistic Regression, Random Forest, metrics
- **imbalanced-learn** — SMOTE, RandomUnderSampler
- **XGBoost** — gradient boosting classifier
- **joblib** — model serialisation

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Practical Exam — Supervised Learning | Fraud Analytics Simulation | November 2026*
# credit-fraud-detection-supervised-learning
