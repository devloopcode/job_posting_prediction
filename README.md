# Job Posting Prediction & Fake Job Detection

A Machine Learning and Data Science project aimed at detecting fake / fraudulent job postings using natural language processing (NLP) and binary classification models.

---

## 📌 Project Overview

Online recruitment fraud is a growing concern for job seekers. Fraudulent job postings can lead to identity theft, financial scams, and wasted effort. This project analyzes a dataset of **17,880 job postings** to uncover patterns that distinguish legitimate job offers from fraudulent ones and builds machine learning models to automatically flag suspicious postings.

---

## 📊 Dataset Summary

The dataset (`fake_job_postings.csv`) contains 17,880 rows and 18 attributes:

- **Target Variable**: `fraudulent` (`0` = Legitimate, `1` = Fraudulent)
- **Class Imbalance**: 
  - **Legitimate (0)**: 17,014 (~95.16%)
  - **Fraudulent (1)**: 866 (~4.84%)
- **Dimensions**: 17,880 rows × 18 columns

### Feature Categories:
- **Textual Features**: `title`, `company_profile`, `description`, `requirements`, `benefits`
- **Categorical & Contextual**: `location`, `department`, `employment_type`, `required_experience`, `required_education`, `industry`, `function`, `salary_range`
- **Binary Metadata Flags**: `telecommuting`, `has_company_logo`, `has_questions`

---

## 🗺️ Project Roadmap

1. **Data Loading & Class Balance Analysis**
   - Inspect dataset structure, check column data types, missing values, and sub-group distributions.
   - Detect duplicate postings across key features.
2. **Text Cleaning & Feature Engineering**
   - Clean textual columns (`title`, `description`, `requirements`, `company_profile`).
   - Consolidate text attributes into a unified text feature vector.
   - Analyze feature missingness as potential signals (e.g., absence of company profile or logo).
3. **Stratified Splitting & Baseline Model**
   - Stratified train/test split to preserve the target class balance.
   - Feature extraction via TF-IDF vectorization coupled with Logistic Regression baseline.
4. **Advanced Modeling & Imbalance Mitigation**
   - Implement Linear SVM / XGBoost classifiers.
   - Apply `class_weight='balanced'` and adjust decision thresholds to optimize Precision and Recall.
5. **Model Interpretation & App Prototyping**
   - Extract top predictive text terms and metadata indicators for fake jobs.
   - Perform misclassification / error analysis.
   - Build an interactive Streamlit application to score real-time job text risk.

---

## 📂 Repository Structure

```
job_posting_prediction/
├── fake_job_postings.csv        # Dataset (17,880 records)
├── job_posting_prediction.ipynb # EDA, preprocessing, and modeling notebook
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Python 3.8+ with the following packages installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Notebook

1. Clone or navigate to the workspace directory:
   ```bash
   cd /Users/macgr/Desktop/projects/job_posting_prediction
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook job_posting_prediction.ipynb
   ```
3. Run all cells to replicate the EDA and preprocessing steps.

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning & NLP**: Scikit-Learn (TF-IDF, Logistic Regression, Linear SVM)
- **Environment**: Jupyter Notebook
