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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, average_precision_score

# Import models
from sklearn.linear_model import LogisticRegression

# Import dalex libraries
import dalex as dx

# set random seed
np.random.seed(123)


# %% ######################## DATA #####################
data = pd.read_csv("../../data/health_data.csv")

# %%
# Check data types
data.dtypes

# Transform data types
data = data.apply(lambda col: col.astype("category") if col.dtype == "object" else col)

# Summarise categorical variables
data.describe(include=["category"])

# Define a mapping for the 'SEX' and 'PMI' columns
mapping = {"male": 0, "female": 1, "yes": 1, "no": 0}

# Apply the mapping to the 'SEX' and 'PMI' columns
data["SEX"] = data["SEX"].map(mapping).astype("int32")
data["PMI"] = data["PMI"].map(mapping).astype("int32")


# %% Split data into X and y
X = data.drop(columns="DAY30")
y = data.DAY30

# %% Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# %% ########################## Modeling pipeline #############################
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Create a list of categorical and numerical features
categorical_features = X.select_dtypes(include=["category"]).columns
numerical_features = X.select_dtypes(include=["int32", "int64", "float64"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)

# %% Train a linear model
# Create model
clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        # Approximate glm parameters from R
        (
            "classifier",
            LogisticRegression(
                fit_intercept=False,
                solver="newton-cg",
                penalty=None,
                max_iter=10000000,
                verbose=1,
            ),
        ),
    ]
)

glm_model = clf.fit(X_train, y_train)
glm_model

# %% Predictions

# Probs on train data
y_train_pred_probs = glm_model.predict_proba(X_train)[:, 1]

# Predict on test data
y_pred_proba = glm_model.predict_proba(X_test)[:, 1]
y_pred = glm_model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
print(cm_df)

# Performance metrics
print(classification_report(y_test, y_pred))

# Compute and print ROC AUC and PR AUC
print("ROC AUC score: {:.3f}".format(roc_auc_score(y_test, y_pred_proba)))
print("PR AUC score: {:.3f}".format(average_precision_score(y_test, y_pred_proba)))

# %% ################################## FAIRNESS ASSESSMENT #####################################
exp_glm = dx.Explainer(
    glm_model,
    data=X_test,
    y=y_test,
    label="glm",
    model_type="classification",
    verbose=True,
)

# %% Performance metrics
exp_glm.model_performance().result

# %% Specify fairness object
fobject_glm = exp_glm.model_fairness(protected=X_test.SEX, privileged=0)

# %% Check fairness metrics
fobject_glm.fairness_check(epsilon=0.8)

# %% Plot fairness metrics
fobject_glm.plot()

# %% ################################## FAIRNESS MITIGATION ######################################


######### Resampling #############
clf_u = copy(clf)
clf_p = copy(clf)

indices_uniform = dx.fairness.resample(X_train.SEX, y_train, verbose=True)
indices_preferential = dx.fairness.resample(
    X_train.SEX,
    y_train,
    type="preferential",  # different type
    probs=y_train_pred_probs,  # requires probabilities
    verbose=False,
)

clf_u.fit(X_train.iloc[indices_uniform, :], y_train.iloc[indices_uniform])
clf_p.fit(X_train.iloc[indices_preferential, :], y_train.iloc[indices_preferential])


exp3 = dx.Explainer(clf_u, X_test, y_test, verbose=False)
exp3.model_performance().result

# %%
exp4 = dx.Explainer(clf_p, X_test, y_test, verbose=False)
exp4.model_performance().result

# %% ######### Reweight #############

weights = dx.fairness.reweight(X_train.SEX, y_train, verbose=True)

clf_weighted = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        # Approximate glm parameters from R
        (
            "classifier",
            LogisticRegression(
                fit_intercept=False,
                solver="newton-cg",
                penalty=None,
                max_iter=10000000,
                verbose=1,
            ),
        ),
    ]
)

kwargs = {clf_weighted.steps[-1][0] + "__sample_weight": weights}

clf_weighted.fit(X_train, y_train, **kwargs)

exp2 = dx.Explainer(clf_weighted, X_test, y_test, verbose=False)
exp2.model_performance().result

# %% ######### ROC pivot #############

exp_roc = copy(exp_glm)

# roc pivot
exp_roc = dx.fairness.roc_pivot(
    exp_roc, protected=X_test.SEX, privileged=0, theta=0.02, verbose=True
)

exp_roc.model_performance().result

# %% ############ COMPARE FAIRNESS METRICS ##################

fobject1 = exp_glm.model_fairness(protected=X_test.SEX, privileged=0, label="glm")
fobject2 = exp2.model_fairness(protected=X_test.SEX, privileged=0, label="weighted")
fobject3 = exp3.model_fairness(protected=X_test.SEX, privileged=0, label="res_unif")
fobject4 = exp4.model_fairness(protected=X_test.SEX, privileged=0, label="res_pref")
fobject5 = exp_roc.model_fairness(protected=X_test.SEX, privileged=0, label="roc")

# plotting
fobject1.plot([fobject2, fobject5, fobject4, fobject3])
