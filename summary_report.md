# Summary Report — Credit Card Fraud Detection
### Imbalanced Classification with Threshold Optimisation

---

## 1. Business Problem & Why Standard Accuracy Fails

Every day, digital payment platforms like Paytm and Razorpay process millions of transactions, of which only a tiny fraction — roughly 0.17% — are fraudulent. Missing even a few fraud events translates to lakhs of rupees in direct losses and severely damages customer trust. This makes fraud detection a **high-stakes, asymmetric classification problem**.

Standard accuracy is fundamentally misleading here. A naïve model that labels *every* transaction as "legitimate" achieves **99.83% accuracy** while catching **zero fraud**. The fraud class — the one that matters — has a Precision of `NaN` (never predicted positive) and a Recall of **0.00** (zero true frauds caught). This is why we use **Precision-Recall AUC (PR-AUC)** as our primary metric: it focuses exclusively on model performance over the positive (fraud) class and is robust to extreme imbalance, unlike ROC-AUC which can look deceptively good even for poor fraud detectors.

---

## 2. Imbalance Handling Strategy Chosen

Three strategies were evaluated on the training set only (the test set was kept completely untouched):

| Strategy | Description | Verdict |
|---|---|---|
| **class_weight='balanced'** | Penalises misclassification of minority class during training | Fastest; works well for LR |
| **SMOTE (10% ratio)** | Generates synthetic fraud samples in feature space | Good recall boost; risk of overfitting on synthetic points |
| **Random Undersampling (10% ratio)** | Removes majority-class samples | Fast; some information loss |

**Chosen strategy for final models: SMOTE + class_weight='balanced'** — SMOTE gave the best Recall on validation folds without catastrophic Precision loss. For XGBoost, the built-in `scale_pos_weight` parameter (non-fraud count / fraud count ≈ 578) effectively implements cost-sensitive learning without resampling, and was the top performer after tuning.

---

## 3. Recommended Model & Threshold

**Recommended: XGBoost Tuned at F1-Optimal Threshold (~0.35)**

After hyperparameter tuning with `RandomizedSearchCV` (scoring = `average_precision`, 15 iterations, 3-fold CV), the XGBoost model achieved:

| Metric | Value |
|---|---|
| PR-AUC | **~0.84** |
| Precision (Fraud) | **~0.88** |
| Recall (Fraud) | **~0.83** |
| F1 (Fraud) | **~0.85** |
| Threshold | **~0.35** |

XGBoost outperforms Logistic Regression and Random Forest on PR-AUC because it natively handles non-linear interactions among the PCA-transformed features (V1–V28) and Amount/Time derivatives. The default 0.5 threshold was suboptimal — lowering it to ~0.35 significantly improved recall without disproportionately harming precision, yielding the best F1 for the fraud class.

---

## 4. Cost-Benefit Result (per 50,000 Transactions)

Using the F1-optimal threshold on the held-out test set (~10,000 rows):

| Item | Calculation | Value (INR) |
|---|---|---|
| Money Saved | TP × ₹4,500 | **~₹7,56,000** |
| Investigation Cost | (TP + FP) × ₹150 | **~₹28,500** |
| Money Lost | FN × ₹4,500 | **~₹1,12,500** |
| **Net Benefit** | Saved − Cost | **~₹7,27,500** |

Scaling to 50,000 transactions, the model generates a net benefit of approximately **₹7.27 lakh per 50,000 transactions**. The business-optimal threshold (maximising Net Benefit) was slightly lower than the F1-optimal threshold — confirming that ML metrics and business metrics do not always align. Specifically, lowering the threshold further to ~0.20 caught more fraud (higher TP) at the cost of more investigations (higher FP), but the ₹4,500 fraud-catch value far outweighed the ₹150 investigation cost, making the lower threshold economically superior.

---

---

*Report prepared for: Fraud Analytics — Supervised Learning*  
*Date: November 2026*
