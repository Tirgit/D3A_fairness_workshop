# D3A Algorithmic Fairness Hackathon

# Introduction
slides uploaded (first 10 mins intro slides)
text
aim of the exercise
fairness: group fairness and which groups are of interest

# Data
description in words
how to download R / Python


install.packages("predtools")
library(predtools)
df <- data(gusto)
?gusto



Variable definitions:

Variable name | Definition | Value
---|---|---
day30 | The 30 Day mortality, the target variable | 0/1
sho | Shock: Killip class 3/4 vs. 1/2 | 0/1
hig | High risk: ANT or PMI | 0/1
dia | Diabetes | 0/1
hyp | Hypotension: Systolic BP<100 mmHg | 0/1
hrt | Heart rate: Pulse>80 bpm | 0/1
ttr | Time to relief of chest pain > 1h | 0/1
sex | Sex (male=0, female=1) | 0/1
Killip | Killip class (1–4): A measure for left ventricular function | 1/2/3/4
age | Age in years | 0-100
ste | Number of leads with ST elevation | 0/1/2/3/4/5/6/7/8/9/10/11/12
pulse | Pulse in beats per minute | 0-200
sysbp | Systolic blood pressure in mmHg | 0-200
ant | Anterior infarct location | 0/1
miloc | MI location: Anterior vs. Inferior vs. Other | 1/2/3
height | Height in cm | 0-200
weight | Weight in kg | 0-200
pmi | Previous myocardial infarction | 0/1
htn | Hypertension history | 0/1
smk | Smoking history: 1 = never; 2 = exsmoker; 3 = current smoker | 1/2/3
pan | Previous angina pectoris | 0/1
fam | Family history of MI | 0/1
prevcvd | Previous CVD | 0/1
prevcabg | Previous CABG | 0/1
regl | xxx | xxx
grpl | xxx | xxx
grps | xxx | xxx
tpa | xxx | xxx
tx | treatment group | 1/2/3




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









