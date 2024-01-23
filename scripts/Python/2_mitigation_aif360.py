## Description: This script is used to create a baseline model for the health data.
##
## Author: Adrian G. Zucco and Tibor V. Varga
## Date Created: 2024-01-22
## Notes:
## Inspired from https://dalex.drwhy.ai/python-dalex-fairness.html

# %% ######################## Import libraries #############################
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy

# Import sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import metrics
from sklearn.metrics import confusion_matrix, classification_report

# Import models
from sklearn.linear_model import LogisticRegression

# Import dalex libraries
import dalex as dx

# Import aif360 libraries


# set random seed
np.random.seed(123)


# %% ######################## DATA #####################
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


# %% ########################## Modeling pipeline #############################
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a list of categorical and numerical features
categorical_features = X.select_dtypes(include=['category']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', 'passthrough', numerical_features)
])

# %% Train a linear model
# Create model
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000, verbose=1))
])

glm_model = clf.fit(X_train, y_train)
y_train_pred_probs = glm_model.predict_proba(X_train)[:,1]

# %%
# Predict on test data
y_pred = glm_model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred))

# %% ################################## FAIRNESS ASSESSMENT #####################################
exp_glm = dx.Explainer(glm_model, data=X_test, y=y_test, 
                       label='glm', 
                       model_type='classification',
                       verbose=True)

# %% Performance metrics
exp_glm.model_performance().result

# %% Specify fairness object

fobject_glm = exp_glm.model_fairness(protected=X_test.SEX, privileged="male")

# %% Check fairness metrics
fobject_glm.fairness_check(epsilon = 0.8)

# %% Plot fairness metrics
fobject_glm.plot()

# %% ################################## FAIRNESS MITIGATION ######################################
