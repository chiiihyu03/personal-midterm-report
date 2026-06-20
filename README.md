# personal-midterm-report
<h1 align="center">🎯 Costa Rican Household Poverty Level Prediction</h1>

<p align="center">
  <i>A Machine Learning Pipeline for Identifying Vulnerable Households using Logistic Regression</i>
</p>

## 📌 Project Overview
The objective of this project is to predict the poverty level of households in Costa Rica. By analyzing various socio-economic features (such as education, housing conditions, and asset ownership), we built a machine learning model to classify households into four vulnerability levels. 

A significant challenge in this dataset is **class imbalance** (extreme poverty cases are the minority). To address this, our pipeline includes stratified sampling, class-weight balancing, and focuses on the `Macro-F1` score for optimization.

## 🏗️ Data Pipeline & Architecture

### 1. Data Preprocessing
* **Categorical Mapping:** Transformed 'yes'/'no' string values into binary numeric values `(1/0)` for features like `dependency`, `edjefe`, and `edjefa`.
* **Missing Value Imputation:**
  * Filled with `0`: `v18q1` (tablets), `v2a1` (monthly rent), `rez_esc` (years behind in school).
  * Filled with `Median`: `meaneduc` (average education) and `SQBmeaned`.

### 2. Feature Engineering (Household Aggregation)
Since the original data contains individual-level records, we aggregated the data at the **Household Level** (`idhogar`) to prevent data leakage and provide a unified household prediction.
* **Aggregated Features include:**
  * `mean`: Average education (`meaneduc`), Age demographics.
  * `max`: Overcrowding (`hacdor`), asset ownership (`v18q1`, `refrig`, `computer`, `television`, `mobilephone`).

### 3. Model Training & Hyperparameter Tuning
* **Algorithm:** Logistic Regression (`saga` solver).
* **Imbalance Handling:** Utilized `class_weight='balanced'` to actively penalize misclassifications of minority classes.
* **Validation Strategy:** `GridSearchCV` with 5-fold Cross-Validation.
* **Evaluation Metric:** Optimized strictly for **Macro-F1 Score**.

## 📊 Final Results

After running the Grid Search pipeline, the model identified the following optimal setup:
* **Optimal Parameters:** `C = 10`, `l1_ratio = 1.0`
* **Validation Set (Macro-F1 Score):** `0.3640`

When evaluated against the **unseen 20% testing set**, the model yielded the following metrics:
* **Overall Accuracy:** `62%`
* **Test Set Macro-F1 Score:** `0.35`
* **Class 1 (Extreme Poverty) Recall:** `23%`

*(Note: While accuracy is heavily influenced by the majority class, the balanced class weights ensure the model makes a concentrated effort to identify Class 1 instances.)*

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the data
Place `train.csv` in the **root of the project folder** (same level as the notebook).

### 4. Launch Jupyter and run the notebook
```bash
jupyter notebook 114AB8059.ipynb
```

Then run all cells from top to bottom (**Kernel → Restart & Run All**).

## 💻 Dependencies

```text
pandas
numpy
scikit-learn
jupyter
```
