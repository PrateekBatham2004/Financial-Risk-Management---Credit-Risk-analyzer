# Credit Risk Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

A robust machine learning pipeline designed to assess credit default risk using the **German Credit Data**. This project focuses on handling severe class imbalance (20:1) and simulating enterprise-scale data environments to minimize financial risk.

## 📌 Overview

In financial risk management, the cost of missing a defaulter (False Negative) is significantly higher than flagging a safe customer (False Positive). This project implements a **Random Forest Classifier** optimized for **Recall** to detect potential defaulters effectively.

To mimic real-world fraud detection scenarios, the original dataset was augmented to **5,000+ records**, introducing a strict **20:1 class imbalance** to stress-test the model's sensitivity.

## 🚀 Key Features

* **Enterprise-Scale Simulation**: Expanded the dataset to **5,000+ records** using statistical resampling to simulate high-volume transaction environments.
* **Imbalance Handling**: Mitigated severe **20:1 class imbalance** using **SMOTE (Synthetic Minority Over-sampling Technique)** to prevent model bias toward the majority class.
* **Feature Engineering**: Processed **18+ financial features**, including *Checking Balance (DM)*, *Credit History*, and *Employment Duration*, to extract meaningful risk signals.
* **Rigorous Validation**:
    * Optimized for **ROC-AUC (0.85)** and **Recall** rather than raw accuracy.
    * Evaluated using **Precision-Recall Curves** to visualize performance on the minority class.
    * Simulated **User Acceptance Testing (UAT)** with 50+ specific test cases.

## 📂 Dataset Details

The project utilizes the **German Credit Data** (originally from UCI Machine Learning Repository).
* **Original Size**: 1,000 records.
* **Augmented Size**: 5,040 records (Simulated).
* **Imbalance Ratio**: 20:1 (95.2% No Default / 4.8% Default).
* **Currency**: Deutsche Mark (DM).

**File:** `data/credit_data_20_1.csv` (Generated via resampling strategies).

## 🛠️ Technologies Used

* **Language**: Python
* **Libraries**: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn` (SMOTE), `matplotlib`
* **Modeling**: Random Forest Classifier, Logistic Regression (for baseline comparison)

## 📊 Model Performance

The model was evaluated specifically on its ability to detect high-risk applicants (Minority Class).

| Metric | Score | Description |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.85** | Strong ability to distinguish between defaulters and non-defaulters. |
| **Precision** | **92%** | High reliability when a default is predicted. |
| **Recall Boost** | **+35%** | Significant improvement in detecting defaulters after applying SMOTE. |

## 🔧 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PrateekBatham2004/Financial-Risk-Management---Credit-Risk-analyzer.git](https://github.com/PrateekBatham2004/Financial-Risk-Management---Credit-Risk-analyzer.git)
    cd Financial-Risk-Management---Credit-Risk-analyzer
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn matplotlib
    ```

3.  **Run the training script:**
    ```bash
    python src/train_model.py
    ```

## 🧪 Testing Strategy

This project adopts a **Quality Assurance** mindset:
* **Data Integrity Tests**: Validated null values and data types for all 18 features before ingestion.
* **Unit Tests**: Checked scaling logic to ensure no data leakage between training and test sets.
* **Stress Testing**: Evaluated model stability against the 20:1 imbalanced dataset to ensure it didn't simply predict "No Default" for all cases.

---
*Developed by Prateek Batham*
