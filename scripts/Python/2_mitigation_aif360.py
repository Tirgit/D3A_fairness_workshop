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
from collections import defaultdict

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

# Import aif360 libraries
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.explainers import MetricTextExplainer

# set random seed
np.random.seed(123)

######################## ACCESORY FUNCTIONS ####################
# %% Function from aif360

def test_aif360(dataset, model, thresh_arr):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]

    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
            dataset,
            dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

        metric_arrs["bal_acc"].append(
            (metric.true_positive_rate() + metric.true_negative_rate()) / 2
        )
        metric_arrs["avg_odds_diff"].append(metric.average_odds_difference())
        metric_arrs["disp_imp"].append(metric.disparate_impact())
        metric_arrs["stat_par_diff"].append(metric.statistical_parity_difference())
        metric_arrs["eq_opp_diff"].append(metric.equal_opportunity_difference())
        metric_arrs["theil_ind"].append(metric.theil_index())

    return metric_arrs


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
print(cm)

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

# %% Transform and prepare data for aif360 data class
preprocessing = preprocessor.fit_transform(X_train)
onehot_categories = preprocessor.named_transformers_["cat"][
    "onehot"
].get_feature_names_out(categorical_features)
X_train_transformed = pd.DataFrame(
    preprocessing, columns=np.concatenate([onehot_categories, numerical_features])
)

preprocessing = preprocessor.fit_transform(X_test)
onehot_categories = preprocessor.named_transformers_["cat"][
    "onehot"
].get_feature_names_out(categorical_features)
X_test_transformed = pd.DataFrame(
    preprocessing, columns=np.concatenate([onehot_categories, numerical_features])
)

X_train_transformed["DAY30"] = y_train.values.astype("int32")
X_test_transformed["DAY30"] = y_test.values.astype("int32")

# %%
attributes_params = dict(protected_attribute_names=["SEX"], 
                         label_names=["DAY30"])

dt_train = BinaryLabelDataset(df=X_train_transformed, **attributes_params)
dt_test = BinaryLabelDataset(df=X_test_transformed, **attributes_params)

# %% In-processing technique - Prejudice Remover (PR) algorithm
PR_model = PrejudiceRemover(eta=25.0, sensitive_attr="SEX", class_attr=0)

# Get scaler from pipeline and transform data
scaler = clf.named_steps["scaler"]
dt_train.features = scaler.transform(dt_train.features)
dt_test.features = scaler.transform(dt_test.features)

# %%
# AttributeError due to use of old numpy version
PR_model = PR_model.fit(dt_train)

# %%
thresh_arr = np.linspace(0.01, 0.50, 50)

val_metrics = test_aif360(dataset=dt_test, model=PR_model, thresh_arr=thresh_arr)
pr_orig_best_ind = np.argmax(val_metrics["bal_acc"])
