## Description: This script is used to create a baseline model for the health data.
##
## Author: Adrian G. Zucco and Tibor V. Varga
## Date Created: 2024-01-22
## Notes:
## Inspired from https://dalex.drwhy.ai/python-dalex-new.html

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import sklearn libraries
from sklearn.model_selection import train_test_split

# Import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV

# Import models
import lightgbm as lgb

# Import dalex libraries
import dalex as dx

# set random seed
np.random.seed(2022024)

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

# Create dataset
train_data = lgb.Dataset(X_train, label=y_train)


# %% ############################## Fit LightGBM model
# Basic parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'verbose': 1
}

# %% Set Random grid search
hyperparameter_grid = {
    'num_leaves': [10, 20, 30, 40],
    'feature_fraction': [0.1, 0.2, 0.3, 0.4, 0.5],
    'bagging_fraction': [0.1, 0.2, 0.3, 0.4, 0.5],
    'max_depth': [5, 10, 15, 20],
    'learning_rate': [0.01, 0.1, 1],
    'min_data_in_leaf': [10, 20, 30, 40, 50],
    'lambda_l1': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'lambda_l2': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
}

# Create a based model
lgbm_hyper = lgb.LGBMClassifier(**params)

# Instantiate the random search model
random_search = RandomizedSearchCV(estimator=lgbm_hyper, param_distributions=hyperparameter_grid, 
                                   n_iter=5, cv=3, n_jobs=-1, verbose=2, scoring='f1')

# Fit the random search to the data
random_search.fit(X_train, y_train)

# Get the best parameters
best_params = random_search.best_params_
best_params

# %%
# Combine best parameters with fixed parameters
params.update(best_params)

# Train final model
lgbm_model = lgb.train(params, train_data, 200)

# %% Predict on test data
y_pred = lgbm_model.predict(X_test)

# Convert probabilities to binary
y_pred = np.where(y_pred > 0.5, 1, 0)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred))

# %% Run Dalex explainer
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
