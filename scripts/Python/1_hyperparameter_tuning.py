## Description: This script is used to create a baseline model for the health data.
##
## Author: Adrian G. Zucco and Tibor V. Varga
## Date Created: 2024-01-22
## Notes:

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Import models
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# Import dalex libraries
import dalex as dx

# set random seed
np.random.seed(123)


# %% Import data
data = pd.read_csv('../../data/health_data.csv')

# %%
# Check data types
data.dtypes

# Transform data types
data = data.apply(lambda col: col.astype('category') if col.dtype == 'object' else col)

# %% Split data into X and y
X = data.drop(columns='DAY30')
y = data.DAY30

# %% Split data into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, stratify=y)


# %% ############################## Fit LightGBM model
# Parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.75,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 1
}

# %% Set hyperparameter tuning


# %%
# Create dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Train model
lgbm_model = lgb.train(params, train_data, 200)

# %% Predict on test data
y_pred = lgbm_model.predict(X_test)

# Convert probabilities to binary
y_pred = np.where(y_pred > 0.5, 1, 0)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %%
exp_gbm = dx.Explainer(lgbm_model, data=X_test, y=y_test, 
                       label='gbm', 
                       model_type='classification',
                       verbose=True)

# %%
exp_gbm.model_performance().result

# %%
fobject = exp_gbm.model_fairness(protected=X_test.SEX, privileged='male')

# %%
fobject.fairness_check(epsilon = 0.8)

# %%
fobject.plot()