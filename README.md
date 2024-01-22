# D3A Conference: Algorithmic Fairness Hackathon

## Introduction
The aim of the following exercise is to get some hands-on experience with implementing algorithmic fairness in a prediction model using a biomedical dataset. Your group will have four distinct tasks (see details below), and you will need to cleverly divide tasks to complete the exercise in time - you will have only 45 minutes! But you don't start from scratch, we have already prepared a dataset and some code to get you started. The key tasks are to assess predictive performance of models across population subgroups defined by protected attributes (sex and immigration status), and to mitigate any potential disparities that you observe.

## Data
For the hackathon we will use the [GUSTO-I trial data](https://pubmed.ncbi.nlm.nih.gov/7882472/) dataset which contains 40,830 observations and 29 variables. The dataset can be used to predict 30 day mortality (DAY30) in patients who have suffered from an acute myocardial infarction.

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
```

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

