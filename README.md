# D3A Conference: Algorithmic Fairness Hackathon

## Introduction
The aim of the following exercise is to get some hands-on experience with implementing algorithmic fairness in a prediction model using a biomedical dataset. Your group will have four distinct tasks (see details below), and you will need to cleverly divide tasks to complete the exercise in time - you will have only 45 minutes! But you don't start from scratch, we have already prepared a dataset and some code to get you started. The key tasks are to assess predictive performance of models across population subgroups defined by protected attributes (sex and immigration status), and to mitigate any potential disparities that you observe.

## Data
For the hackathon we will use the [GUSTO-I trial data](https://pubmed.ncbi.nlm.nih.gov/7882472/) dataset which contains 40,830 observations and 29 variables. The dataset can be used to predict 30 day mortality (DAY30) in patients who have suffered from an acute myocardial infarction.

[Detailed information about the data here](data/Dataset.md)

## Prediction
slip train test
define a ML for predicting outcome (binary classification)
confusion matrix
define metrics (F1 / AUPRC)
stratified metrics (output F1 / AUPRC for men vs women)

## Fairness
The sensitive or protected attribute in the dataset is sex. We would like to achieve prediction models that perform equally between men and women. 

recap on confusion matrix
fairness metrics: Equalized Odds and Predictive Rate Parity are two most relevant (but there are more)
how to achieve fairness / mitigate bias
-> preprocessing
-> algorithmic tuning
-> postprocessing
https://journal.r-project.org/articles/RJ-2022-019/


## Groups
1. Improve the model (e.g., feature selection, from GLM to ML), but no fairness constraints
2. Improve the model: Pre-processing
3. Improve the model: Optimization
4. Improve the model: Post-processing

Upload results to:
https://docs.google.com/presentation/d/1D2Fc44sKXB3b5-tT3yvADzPGcQFUQ4AcVz6VDRiw15U/edit?usp=sharing


## Discussion
upload results to : xxxxxx google slides
results from performance metrics and fairness metrics

## How to cite

TODO: Generate Zenodo DOI for this workshop







