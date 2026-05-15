# SDG 1: No Poverty.

# 1.Data Loading and Initial Exploratory Analysis

import pandas as pd

# Load data from the local directory
file_path = 'train.csv'

train = pd.read_csv(file_path)
print("資料大小:", train.shape)
train.head()


# 2. Data Cleaning and Feature Preprocessing

import numpy as np

mapping = {"yes": 1, "no": 0}

cols_to_fix = ['dependency', 'edjefe', 'edjefa']
for col in cols_to_fix:
    train[col] = train[col].replace(mapping).astype(float)

print("✅ Categorical mapping complete: 'yes/no' converted to 1/0.")


train['v18q1'] = train['v18q1'].fillna(0)


train['v2a1'] = train['v2a1'].fillna(0)


train['rez_esc'] = train['rez_esc'].fillna(0)


train['meaneduc'] = train['meaneduc'].fillna(train['meaneduc'].median())
train['SQBmeaned'] = train['SQBmeaned'].fillna(train['SQBmeaned'].median())

print("✅ Missing value imputation complete.")


print("\nRemaining Missing Values Check:")
print(train[['v18q1', 'v2a1', 'rez_esc', 'meaneduc']].isnull().sum())

# 3.Household-Level Data Aggregation

# Define aggregation rules based on feature characteristics
aggregation_rules = {
    'meaneduc': 'mean',   # Average education level of the household
    'hacdor': 'max',      # Overcrowding indicator
    'v18q1': 'max',       # Number of tablets
    'refrig': 'max',      # Presence of a refrigerator
    'computer': 'max',    # Presence of a computer
    'television': 'max',  # Presence of a television
    'mobilephone': 'max', # Presence of a mobile phone
    'qmobilephone': 'max',# Total count of mobile phones
    'age': ['mean', 'min', 'max'] # Age demographics (Avg, Min, Max)
}

# Execute aggregation grouped by the household ID (idhogar)
df_household = train.groupby('idhogar').agg(aggregation_rules)

# Flatten the multi-level columns created by aggregation
df_household.columns = ['_'.join(col).strip() for col in df_household.columns.values]

# Join the Target variable back to the aggregated data
target = train.groupby('idhogar')['Target'].first()
df_final = df_household.join(target)

print(f"Aggregation complete. Individual rows: {len(train)} -> Household rows: {len(df_final)}")

# 4.Model Training and Hyperparameter Tuning

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Prepare feature matrix (X) and target vector (y)
X = df_final.drop('Target', axis=1)
y = df_final['Target']

# Splitting data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

logreg = LogisticRegression(solver='saga', max_iter=1000, class_weight='balanced')

param_grid = {
    'C': [0.1, 1, 10],
    'l1_ratio': [0.0, 1.0]
}

# Perform Grid Search focusing on Macro-F1 score
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train, y_train)

# Final Results Output
print(f"Optimal Parameters: {grid_search.best_params_}")
print(f"Validation Set F1-Score: {grid_search.best_score_:.4f}")

# Model Evaluation on Test Set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\n--- Classification Report (Class Weight Balanced) ---")
print(classification_report(y_test, y_pred))

