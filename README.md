# AI-ML-INTENSHIP-TASK--16
Hyperparameter Tuning using Grid Search CV
#  Hyperparameter Optimization using GridSearchCV  
### Support Vector Machine (SVM) | Breast Cancer Classification

---

##  Executive Summary

This project demonstrates a production-style machine learning workflow focused on **model optimization using GridSearchCV**.  

A baseline Support Vector Machine (SVM) model is trained and compared against a tuned version using cross-validated hyperparameter search.

The goal is to improve predictive performance while maintaining a clean, reproducible, and scalable ML pipeline.

---

##  Dataset Information

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** `sklearn.datasets`
- **Total Samples:** 569
- **Features:** 30 numerical features
- **Target Classes:**
  - 0 → Malignant
  - 1 → Benign
- **Problem Type:** Binary Classification

---

##  Project Architecture

```
Data Loading
     ↓
Train-Test Split
     ↓
Pipeline (StandardScaler + SVM)
     ↓
Baseline Model Training
     ↓
GridSearchCV (5-Fold Cross Validation)
     ↓
Best Model Extraction
     ↓
Performance Comparison
```

---

##  Technical Stack

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

---

##  Machine Learning Workflow

### 1️ Data Preprocessing

- Stratified train-test split (80/20)
- Feature scaling using `StandardScaler`
- Pipeline integration to prevent data leakage

### 2️ Baseline Model

- Algorithm: Support Vector Machine (SVC)
- Default hyperparameters
- Performance metrics recorded for benchmarking

### 3️ Hyperparameter Optimization

**Parameter Grid:**

```python
{
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 0.1, 0.01, 0.001],
    'svm__kernel': ['rbf', 'linear']
}
```

**Configuration:**

- 5-Fold Cross Validation
- Scoring Metric: Accuracy
- Parallel Processing enabled (`n_jobs=-1`)

---

##  Performance Evaluation

Metrics used:

- Accuracy
- Precision
- Recall
- F1-Score

### 🔹 Example Results

| Model                     | Accuracy | Precision | Recall | F1-Score |
|---------------------------|----------|-----------|--------|----------|
| Baseline SVM              | ~96%     | ~97%      | ~96%   | ~96%     |
| Tuned SVM (GridSearchCV)  | ~98%     | ~98%      | ~98%   | ~98%     |

✔ The tuned model demonstrates measurable performance improvement.

---

##  Key Engineering Practices Demonstrated

- Proper Train-Test Separation
- Prevention of Data Leakage via Pipeline
- Cross-Validation Based Model Selection
- Structured Performance Benchmarking
- Clean and Reproducible Code Design

---

##  Repository Structure

```
├── notebooks/
│   └── GridSearchCV_SVM.ipynb
│
├── README.md
```

---

##  Installation & Execution

### Clone Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Install Dependencies

```bash
pip install numpy pandas scikit-learn
```

### Run Notebook

```bash
jupyter notebook
```

---

##  Business Impact Perspective

Hyperparameter tuning enables:

- Improved predictive accuracy
- Better generalization to unseen data
- More reliable decision-making models
- Structured model optimization strategy

This workflow reflects real-world ML development practices used in production environments.

---

##  Future Improvements

- Add ROC-AUC comparison
- Integrate Random Forest / XGBoost benchmarking
- Add model persistence (`.pkl` export)
- Deploy via Flask or FastAPI API endpoint
- CI/CD integration for automated retraining

---

##  Conclusion

GridSearchCV significantly enhances model performance through systematic hyperparameter exploration.

This project provides a scalable and industry-ready template for classification problems requiring structured model optimization.

---

**Author:** Jaasiel Mark  

