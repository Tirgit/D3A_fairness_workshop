# Dataset

The dataset for this project is from GUSTO-I trial:

> Lee, Kerry L., Lynn H. Woodlief, Eric J. Topol, W. Douglas Weaver, Amadeo Betriu, Jacques Col, Maarten Simoons, Phil Aylward, Frans Van de Werf, and Robert M. Califf. "Predictors of 30-day mortality in the era of reperfusion for acute myocardial infarction: results from an international trial of 41 021 patients." Circulation 91, no. 6 (1995): 1659-1668.

The data has been downloaded from the publicly available repository of the [predtools R package](https://github.com/resplab/predtools/tree/master/data). We have transformed the data into a CSV file for easier use.

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
TX | treatment group SK vs tPA vs SK+tPA | factor
