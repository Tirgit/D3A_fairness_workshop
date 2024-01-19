# D3A Conference: Algorithmic Fairness Hackathon

# Introduction
The aim of the following exercise is to get some hands-on experience with implementing algorithmic fairness in a prediction model using a biomedical dataset. Your group will have four distinct tasks (see details below), and you will need to cleverly divide tasks to complete the exercise in time - you will have only 45 minutes! But you don't start from scratch, we have already prepared a dataset and some code to get you started. The key tasks are to assess predictive performance of models across population subgroups defined by protected attributes (sex and immigration status), and to mitigate any potential disparities that you observe.

# Data
For the hackathon we will use the *gusto* dataset which contains 40,830 observations and 29 variables. The dataset can be used to predict 30 day mortality (DAY30) in patients who have suffered from an acute myocardial infarction. The data set consists of a subset of the [GUSTO-I trial data](https://pubmed.ncbi.nlm.nih.gov/7882472/).

You can load data data by running the following code:
df <- read_csv("../data/health_data.csv")

The dataset contains the following variables:

Variable name | Definition | Type
---|---|---
DAY30 | The 30 Day mortality, the target variable | binary
SHO | Shock: Killip class 3/4 vs. 1/2 | binary
HIG | High risk: ANT or PMI | binary
DIA | Diabetes | binary
HYP | Hypotension: Systolic BP<100 mmHg | binary
HRT | Heart rate: Pulse>80 bpm | binary
TTR | Time to relief of chest pain > 1h | binary
SEX | Sex (male=0, female=1) | factor
KILLIP | Killip class (1–4): A measure for left ventricular function | factor
AGE | Age in years | numeric (19-110)
STE | Number of leads with ST elevation | numeric (0-11)
PULSE | Pulse in beats per minute | numeric (0-246)
SYSBP | Systolic blood pressure in mmHg | numeric (0-280)
ANT | Anterior infarct location | binary
MILOC | MI location: Anterior vs. Inferior vs. Other | factor
HEIGHT | Height in cm | numeric (140-212.5)
WEIGHT | Weight in kg | numeric (36-213)
PMI | Previous myocardial infarction | binary
HTN | Hypertension history | binary
SMK | Smoking history: 1 = never; 2 = exsmoker; 3 = current smoker | factor
PAN | Previous angina pectoris | binary
FAM | Family history of MI | binary
PREVCVD | Previous CVD | binary
PREVCABG | Previous CABG | binary
REGL | REGL protein | numeric (1-16)
GRPL | GRPL protein | numeric (1-48)
GRPS | GRPS protein | numeric (1-121)
TPA | presence of TPA | binary
TX | treatment group SK vs tPA vs SK+tPA | factor




# Prediction
slip train test
define a ML for predicting outcome (binary classification)
confusion matrix
define metrics (F1 / AUPRC)
stratified metrics (output F1 / AUPRC for men vs women)

# Fairness
The sensitive or protected attribute in the dataset is sex. We would like to achieve prediction models that perform equally between men and women. 

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

Upload results to:
https://docs.google.com/presentation/d/1D2Fc44sKXB3b5-tT3yvADzPGcQFUQ4AcVz6VDRiw15U/edit?usp=sharing


# Discussion
upload results to : xxxxxx google slides
results from performance metrics and fairness metrics









