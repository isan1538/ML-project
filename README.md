# Fraud Detection in E-Commerce Using Machine Learning

This repository contains a complete machine learning pipeline for detecting fraudulent transactions in large-scale e-commerce systems. The project evaluates multiple machine learning algorithms, performs extensive feature engineering, and applies dimensionality-reduction techniques using PCA and t-SNE. All methods and results are fully documented in the accompanying **Final_ML_Report.pdf** and implemented in two Jupyter notebooks: **Project.ipynb** and **Project_PCA.ipynb**.

---

## 📌 Project Summary

Financial fraud poses a significant challenge in online marketplaces due to highly imbalanced data, geographical diversity, and evolving attacker strategies. This project explores fraud detection on a realistic synthetic dataset that includes:

* **~300,000 transactions** across
* **6,000 users**, each generating 40–60 purchases
* **Fraud rate of ~2%**, closely reflecting real-world systems

The goal is to classify each transaction as **fraudulent** or **legitimate** using engineered behavioral, temporal, and geographic features.

---

## 📂 Repository Structure

```
├── Project.ipynb                # Full ML pipeline: preprocessing, modeling, evaluation
├── Project_PCA.ipynb            # PCA & t-SNE dimensionality reduction experiments
├── Final_ML_Report.pdf          # Full technical report and results
├── data/                        # Optional dataset folder
└── README.md                    # Project documentation
```

---

## 🤖 Machine Learning Models

The following supervised models were trained and evaluated:

* **K-Nearest Neighbors (KNN)**
* **Support Vector Machine (RBF SVM)**
* **Multilayer Perceptron (Neural Network)**
* **Bagging Ensemble**
* **XGBoost Classifier**

Evaluation metrics emphasize the minority fraud class:

* Precision
* Recall
* F1-score
* ROC–AUC

---

## 📊 Key Results (Full Feature Set)

| Model     | AUC        | Fraud Recall | Fraud F1   |
| --------- | ---------- | ------------ | ---------- |
| KNN       | 0.8545     | 0.5287       | 0.6676     |
| SVM (RBF) | 0.9037     | 0.5688       | 0.7158     |
| NeuralNet | 0.9733     | 0.7337       | 0.8196     |
| Bagging   | 0.9504     | 0.7587       | 0.8393     |
| XGBoost   | **0.9739** | **0.7648**   | **0.8321** |

### 🏆 Best Model: **XGBoost**

XGBoost achieved the highest fraud recall and AUC, demonstrating strong sensitivity to rare fraudulent transactions.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/isan1538/ML-project.git
cd fraud-detection-ml
```

### 2. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open:

* `Project.ipynb` for the main analysis
* `Project_PCA.ipynb` for PCA/t-SNE experiments

---

## ✔ Key Insights

* Fraud detection performance improves significantly with behavioral and geographical features.
* Ensemble methods (XGBoost, Bagging) outperform distance- and margin-based models.
* PCA maintains strong predictive accuracy while reducing feature complexity.
* Neural networks provide competitive results across both full-feature and PCA-reduced datasets.

---

## 👥 Authors

* **Ehsan Zeraatkar** — Ph.D. Student, Computer Science
* **Maria Sultana** — Ph.D. Student, Materials Science, Engineering & Commercialization

---
