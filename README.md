# D3A Conference: Algorithmic Fairness Hackathon

## Introduction
The aim of the following exercise is to get some hands-on experience with implementing algorithmic fairness in a prediction model using a biomedical dataset. Your group will have four distinct tasks (see details below), and you will need to cleverly divide tasks to complete the exercise in time - you will have only 45 minutes! But you don't start from scratch, we have already prepared a dataset and some code to get you started. The key tasks are to assess predictive performance of models across population subgroups defined by protected attributes (sex and immigration status), and to mitigate any potential disparities that you observe.

## Data
For the hackathon we will use the [GUSTO-I trial data](https://pubmed.ncbi.nlm.nih.gov/7882472/) dataset which contains 40,830 observations and 28 variables. The dataset can be used to predict 30 day mortality (DAY30) in patients who have suffered from an acute myocardial infarction.

[Detailed information about the data here](data/Dataset.md)

## Setting up your environment

### Python users

Instructions to create a conda environment are in the [conda environment file](scripts/Python/hackathon_env.yml) and necessary packages in the `pip` [requirements file](scripts/Python/requirements.txt). You can create the environment with the necessary packages executing the following commands in your terminal (assuming you are at the root directory of this repository):

```bash
  conda env create -f ./scripts/Python/hackathon_env.yml
  conda activate hackathon_env
  pip install -r ./scripts/Python/requirements.txt
```

Execution of the Python scripts, cell by cell, has been tested in [Visual Studio Code](https://code.visualstudio.com/)

### R users

To work with your current installation of R (and Rstudio), uncomment the first lines in the scripts to install the necessary packages. We suggest to install the [pak library](https://pak.r-lib.org/) first to manage the installation of the other packages. You can install it with the following command:

```R
install.packages("pak", repos = "https://r-lib.github.io/p/pak/devel/")
pak::pkg_install(c("skimr", "tidyverse", "tidymodels", "rsample", "lightgbm", "parsnip", "bonsai", "tune", "yardstick", "DALEX", "DALEXtra", "fairmodels", "ggplot2", "pROC", "gbm"))
```

## Prediction
We recommend that you start with a baseline model (GLM) in the *0_baseline_fairness_bias* script, and then assess model performances depending on your task in your group in the *0_baseline_fairness_bias* script or the other supplied scripts. Use a metrics of prediction of your choice. This can be a single confusion matrix metric, or a more sophisticated metric, such as ROC AUC, PRAUC, F1-score, etc. We suggest you stick with one metric and use it throughout the hackathon. The standard functions (*model_performance* in R, and *classification_report* in Python) will output precision, recall, F1-score, accuracy, and ROC AUC. Feel free to implement other metrics.

## Fairness
The sensitive or protected attribute in the dataset is sex. We would like to achieve prediction models that perform equally between men and women. Many metrics exist to assess algorithmic group fairness, and the most common ones use [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) metrics. In short, the aim is usually to choose a relevant confusion matrix metric and try to equalize it in the pre-selected population subgroups - in this case, by sex. You will see that the DALEX/fairmodels pipeline that are implemented in the scripts will have a standard set of metrics that you can use to assess fairness.

These metrics are: 
- Accuracy equality ratio (TP+TN)/(TP+TN+FP+FN)
- Equal opportunity ratio TP/(TP+FN)
- Predictive equality ratio FP/(FP+TN)
- Predictive parity ratio TP/(TP+FP)
- Statistical parity ratio (TP+FP)/(TP+FP+TN+FN)

Whereas in the AIF360 pipeline, the following metrics are available:

- Statistical parity difference
- Equal opportunity difference
- Average odds difference 1/2[(FPR<sub>D=unprivileged</sub>−FPR<sub>D=privileged</sub>)+(TPR<sub>D=unprivileged</sub>−TPR<sub>D=privileged</sub>))]
- Disparate impact
- [Theil index](https://en.wikipedia.org/wiki/Theil_index)

Try to think about which metric is the most relevant for this use case.

The *0_baseline_fairness_bias* script (and the *2_mitigation_aif360* script for Python users) contains various strategies to mitigate bias. Bias mitigation strategies can be divided into three categories: pre-processing, in-processing and post-processing. 
- Pre-processing strategies are applied before the model is trained,
- In-processing strategies are applied during the training of the model, and
- Post-processing strategies are applied after the model is trained. 
R users will focus on pre-processing and post-processing, while Python users can also explore in-processing strategies.

## The Task
You are divided in groups of 4-5 people. You will have 45 minutes to complete the following tasks, and we suggest that you divide the tasks between the group members and work in parallel and communicate, communicate, communicate.
The tasks are:
1. Record performance of baseline GLM model
2. Try to improve the model (e.g., via feature selection, implementing more sophisticated ML approach, parameter tuning). We implemented a LightGBM algorithm, but you are free to try other algorithms. Do NOT implement any bias mitigation strategies yet, only focus on obtaining a better model.
3. Improve the model via bias mitigation in the pre-processing stage
4. Improve the model via bias mitigation in the in-processing stage (Python only)
5. Improve the model via bias mitigation in the post-processing stage

Record the overall performance of the model, the stratified performance of the model (men vs. women), and the fairness metric of your choice.

> Upload results and your key learnings to:
https://docs.google.com/presentation/d/1D2Fc44sKXB3b5-tT3yvADzPGcQFUQ4AcVz6VDRiw15U/edit?usp=sharing

## How to cite code from this repository
TODO: Generate Zenodo DOI for this workshop
