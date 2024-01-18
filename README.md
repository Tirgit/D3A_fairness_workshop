# D3A Algorithmic Fairness Hackathon

# Introduction
slides uploaded (first 10 mins intro slides)
text
aim of the exercise
fairness: group fairness and which groups are of interest

# Data
description in words
how to download R / Python

Variable definitions:

Variable name | Definition | Value
---|---|---
Juicy Apples | 1.99 | 739
day30 | The 30 Day mortality, the target variable | 0/1
sho | Shock: Killip class 3/4 vs. 1/2 | 0/1



AGE - Age in years
A65 - Age >65 years (0/1)
SEX Gender (male=0, female=1)
KILLIP Killip class (1–4): A measure for left ventricular function
DIA Diabetes (0/1)
HYP Hypotension: Systolic BP<100 (0/1)
HRT Heart rate: Pulse>80 (“tachycardia,” 0/1)
ANT Anterior infarct location (0/1)
PMI Previous myocardial infarction (0/1)
HIG High risk: ANT or PMI (0/1)
HEI Height in cm
WEI Weight in kg
SMK Smoking (1 = never; 2 = exsmoker; 3 = current smoker)
HTN Hypertension history (0/1)
LIP Lipids: Hypercholesterolaemia (0/1)
PAN Previous angina pectoris (0/1)
FAM Family history of MI (0/1)
STE ST elevation on ECG: Number of leads
ST4 ST elevation on ECG: >4 leads (0/1)
TTR Time to relief of chest pain > 1 h (0/1)

# Prediction
slip train test
define a ML for predicting outcome (binary classification)
confusion matrix
define metrics (F1 / AUPRC)
stratified metrics (output F1 / AUPRC for men vs women)

# Fairness
recap on confusion matrix
fairness metrics: Equalized Odds and Predictive Rate Parity are two most relevant (but there are more)
how to achieve fairness / mitigate bias
-> preprocessing
-> algorithmic tuning
-> postprocessing
https://journal.r-project.org/articles/RJ-2022-019/


# Groups
1. Improve the model (e.g., feature selection, from GLM to ML), but no fairness constraints
2. Improve the model: Pre-processing
3. Improve the model: Optimization
4. Improve the model: Post-processing

# Discussion
upload results to : xxxxxx google slides
results from performance metrics and fairness metrics









