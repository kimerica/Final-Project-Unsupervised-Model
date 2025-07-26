# Credit Card Fraud Detection (Unsupervised & Supervised ML)

This project detects fraudulent transactions in a real-world credit card dataset using both unsupervised and supervised machine learning techniques. The primary focus is on identifying fraud without labels using anomaly detection, then comparing performance with a supervised model.

---

## Dataset

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description:** 284,807 credit card transactions with 492 labeled as fraudulent (highly imbalanced)
- **Features:** PCA-transformed anonymized features + `Time`, `Amount`, and `Class` (0 = non-fraud, 1 = fraud)

---

## Tools & Libraries

- **Language:** Python  
- **Libraries:** `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `numpy`

---

## Models Used

### Unsupervised Learning:
- **Isolation Forest**
- **Local Outlier Factor (LOF)**

### Supervised Learning (Baseline):
- **Gradient Boosting Classifier** (XGBoost)

---

## Evaluation Metrics

- **Precision**
- **Recall**
- **Precision-Recall AUC** (Area Under Curve) <- Chosen because dataset is highly imbalanced and standard accuracy is misleading.

---

## Workflow

1. Load and scale the credit card transaction data
2. Apply Isolation Forest and LOF to detect anomalies
3. Evaluate performance using precision-recall curves
4. Train a supervised model (XGBoost) as a baseline
5. Compare results and visualize outcomes

---

## Results Summary

- **Unsupervised models** detected fraud without seeing labels:
  - Isolation Forest performed better than LOF
- **XGBoost** showed the highest overall performance but requires labeled data
- **Precision-Recall AUC** was the most informative metric due to class imbalance


---

## How to Run

1. Clone this repo
2. Download the dataset from Kaggle and place `creditcard.csv` in the same folder
3. Install dependencies: `pip install pandas scikit-learn xgboost matplotlib seaborn`
4. Run the notebook: `credit-card-fraud-unsupervised.ipynb`

---

## Author

**Erica Kim**  
Master’s in Data Science – University of Colorado Boulder  
[GitHub Profile](https://github.com/kimerica)
