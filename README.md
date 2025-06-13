#  Fraud Detection Using Machine Learning on the IEEE-CIS Dataset

This project presents a comparative analysis of multiple machine learning algorithms applied to the **IEEE-CIS Fraud Detection** dataset. It explores both performance and interpretability aspects using a range of models and Explainable AI (XAI) techniques.

---

##  Research Questions

- **RQ1:** How do traditional ML models perform on a highly imbalanced, real-world fraud detection dataset compared to their performance in published studies?
- **RQ2:** Which model offers the best trade-off between predictive performance and interpretability?

---

##  Related Work

Numerous works have benchmarked ML models for fraud detection, but most focus solely on improving accuracy and neglect **explainability**. Prior studies (Dornadula & Geetha, Varmedja et al., Thennakoon et al.) have evaluated models like Random Forests, MLPs, and SVMs but rarely incorporated **XAI** tools like SHAP or LIME. Our work fills this gap by combining high-performing models with interpretability techniques for financial accountability.

---

##  Methodology

###  Dataset
- **Source:** [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)
- **Samples:** ~590,000 transactions
- **Target:** `isFraud` (binary classification)

### Preprocessing
- Imputed missing values
- Scaled numerical features
- Encoded categorical variables
- Used **SMOTE** and **class weighting** to address class imbalance

###  Models Evaluated
- Logistic Regression
- Random Forest
- Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)
- XGBoost

###  Hyperparameter Tuning
Used **Grid Search** or **Random Search** with 5-fold cross-validation.

###  Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

---

##  Explainability Techniques

To improve transparency and trust, the following **XAI tools** were used:
- **SHAP** – Feature impact analysis
- **LIME** – Local prediction interpretability
- **PDP / ICE** – Feature effect visualizations
- **PFI** – Permutation-based importance
- **EDA** – Feature distribution & correlation analysis

---

##  Results Snapshot

| Model                  | Accuracy | Class | Precision | Recall | F1-Score |
|------------------------|----------|-------|-----------|--------|----------|
| **Random Forest**      | 99.56%   | 0     | 0.99      | 1.00   | 0.99     |
|                        |          | 1     | 0.78      | 0.47   | 0.58     |
| **XGBoost**            | 99.31%   | 0     | 0.99      | 1.00   | 0.99     |
|                        |          | 1     | 0.78      | 0.47   | 0.58     |
| **Decision Tree**      | 98.32%   | 0     | 0.96      | 0.96   | 0.96     |
|                        |          | 1     | 0.57      | 0.50   | 0.53     |
| **Support Vector Machine** | 93.53% | 0     | 0.98      | 0.95   | 0.97     |
|                        |          | 1     | 0.11      | 0.22   | 0.14     |
| **Logistic Regression**| 93.00%   | 0     | 0.99      | 0.94   | 0.96     |
|                        |          | 1     | 0.18      | 0.62   | 0.28     |
| **Naive Bayes**        | 58.00%   | 0     | 0.99      | 0.58   | 0.73     |
|                        |          | 1     | 0.04      | 0.78   | 0.08     |

---

##  Key Findings

- **Random Forest** and **XGBoost** offered the best overall performance in terms of accuracy and balanced precision/recall for the fraud class.
- **Naive Bayes** and **Logistic Regression** struggled with the imbalanced class, showing high false negatives.
- **SVM** had difficulty classifying fraud (Class 1) due to class imbalance.
- **XAI tools** like SHAP and LIME helped expose feature contributions and supported model debugging and trustworthiness.

---

##  Conclusion

This study confirms that while tree-based models (Random Forest, XGBoost) provide excellent fraud detection performance, they must be complemented with interpretability tools to be viable in production. By integrating **XAI**, we bridge the gap between **performance** and **trust**, which is essential in financial applications.

---


