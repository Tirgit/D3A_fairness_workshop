## Description: This script is used to create a baseline GLM model for the health data.
##
## Author: Adrian G. Zucco and Tibor V. Varga
## Date Created: 2024-01-18
## Notes:
## Inspiration from: https://www.kaggle.com/code/athosdamiani/lightgbm-with-tidymodels
## Fairness tutorial: https://cran.r-project.org/web/packages/fairmodels/vignettes/Advanced_tutorial.html
## Bias mitigation: https://towardsdatascience.com/fairmodels-lets-fight-with-biased-machine-learning-models-f7d66a2287fc

## ------------------------------------
# Install list of libraries

# Install all the libraries with pak
# install.packages("pak", repos = "https://r-lib.github.io/p/pak/devel/")
# pak::pkg_install(c("pROC", "tidyverse", "tidymodels", "ggplot2", "fairmodels", 
# "parsnip", "DALEX", "tune", "yardstick", "DALEX", "DALEXtra"))

# Load libraries
library(pROC)
library(tidyverse)
library(tidymodels)
library(parsnip)
library(tune)
library(yardstick)
library(DALEX)
library(DALEXtra)
library(fairmodels)
library(pROC)
library(ggplot2)
library(gbm)

# Install AIF360 from GitHub
devtools::install_github("Trusted-AI/AIF360/aif360/aif360-r")
library(aif360)

####################################################################
####################################################################
##################### INITIAL PREDICTIONS ##########################
####################################################################
####################################################################

# LOAD DATA
rawdata <- read_csv("../../data/health_data.csv")

# convert DAY30 to factor
rawdata$DAY30 <- factor(rawdata$DAY30, levels = c(0, 1), labels = c("no", "yes"))

# convert all character variables to factors
data <- rawdata %>%
  mutate_if(is.character, as.factor)

# DATA PARTITIONING
set.seed(23456)
split_data <- initial_split(data, prop = 0.70, strata = DAY30)
training <- training(split_data)
testing <- testing(split_data)

# DEFINE LOGISTIC REGRESSION MODEL
model <- logistic_reg() |> 
  set_engine("glm")

# DEFINE RECIPE
rec <- recipe(DAY30 ~ ., data = training)

# DEFINE WORKFLOW
wkflow <- workflow() |> 
  add_model(model) |> 
  add_recipe(rec)

# TRAINING MODEL
model_fit <- wkflow |> 
  fit(data = training)

# ASSESS MODEL ON TEST DATA
preds <- model_fit |> 
  augment(new_data = testing)

# DRAW ROC CURVE
result.roc <- roc(testing$DAY30, preds$.pred_yes) 
ggroc(result.roc, alpha = 0.5, colour = "red", linetype = 1, size = 1) + 
  theme_minimal() + 
  ggtitle("ROC curve - logistic regression") + 
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), color="grey", linetype="dashed")

# CONFUSION MATRIX
conf_mat(preds, truth = DAY30, estimate = .pred_class)


####################################################################
####################################################################
###################### FAIRMODELS R PACKAGE ########################
##################### GROUP FAIRNESS METRICS #######################
####################################################################
####################################################################

# define outcome (outcome set to 0 and 1, it needs to be numeric)
y_training <- as.numeric(training$DAY30) -1
y_testing <- as.numeric(testing$DAY30) -1

# FAIRNESS CHECK
explainer_lm <- explain(model_fit, data = testing[,-1], y = y_testing)
fobject <- fairness_check(explainer_lm,
                          protected = testing$SEX,
                          privileged = "male",
                          epsilon = 0.8,
                          label = c("GLM"))

# BASIC METRICS
model_performance(explainer_lm)

# CHECK IF FAIRNESS CHECK PASSED FOR 5 SELECTED METRICS
fobject

# NUMERIC FAIRNESS METRICS
fobject$groups_data$workflow

# VISUALIZE FAIRNESS CHECK
# FIRST, RAW METRICS
plot(metric_scores(fobject))
# THEN, RELATIVE METRICS
plot(fobject)

# PROBABILITIES
plot_density(fobject)


####################################################################
####################################################################
####################### BIAS MITIGATION ############################
############### FAIRMODELS & DALEX R PACKAGES ######################
######################## PRE-PROCESSING ############################
####################################################################
####################################################################

####################################################################
######################## REWEIGHTING ###############################
####################################################################

# calculation weights
weights <- reweight(protected = testing$SEX, y = as.numeric(testing$DAY30)-1)

# convert outcome to numeric
testing$DAY30 <- as.numeric(testing$DAY30)-1

# run GBM model as a new baseline
set.seed(1356)
gbm_model <- gbm(DAY30 ~. ,
                 data = testing,
                 distribution = "bernoulli")

# run reweighted GBM model
gbm_model_weighted <- gbm(DAY30 ~. ,
                 data = testing,
                 weights = weights,
                 distribution = "bernoulli")

# explain baseline and reweighted models
gbm_explainer <- explain(gbm_model,
                           data = testing[,-1],
                           y = testing$DAY30)

gbm_explainer_w <- explain(gbm_model_weighted,
                           data = testing[,-1],
                           y = testing$DAY30)

# model performance
model_performance(gbm_explainer)
model_performance(gbm_explainer_w)

# fairness check
fobject <- fairness_check(fobject, gbm_explainer, gbm_explainer_w,
                          protected = testing$SEX,
                          privileged = "male",
                          label = c("GBM", "GBM_weighted"))

# visualize fairness check
plot(fobject)


####################################################################
######################### RESAMPLING ###############################
####################################################################

# to obtain probabilities we will use simple GLM
probs <- glm(DAY30 ~., data = testing, family = binomial())$fitted.values

# calculate indeces for resampling in two ways (uniform and preferential)
uniform_indexes      <- resample(protected = testing$SEX,
                                 y = testing$DAY30)
preferential_indexes <- resample(protected = testing$SEX,
                                 y = testing$DAY30,
                                 type = "preferential",
                                 probs = probs)

# run GBM model on resampled data in two ways (uniform and preferential)
set.seed(5777)
gbm_model_u     <- gbm(DAY30 ~. ,
                     data = testing[uniform_indexes,],
                     distribution = "bernoulli")
gbm_model_p     <- gbm(DAY30 ~. ,
                     data = testing[preferential_indexes,],
                     distribution = "bernoulli")

# explain resampled models (uniform and preferential)
gbm_explainer_u <- explain(gbm_model_u,
                           data = testing[,-1],
                           y = testing$DAY30,
                           label = "gbm_uniform")

gbm_explainer_p <- explain(gbm_model_p,
                           data = testing[,-1],
                           y = testing$DAY30,
                           label = "gbm_preferential")

# model performance
model_performance(gbm_explainer_u)
model_performance(gbm_explainer_p)

# fairness check
fobject <- fairness_check(fobject, gbm_explainer_u, gbm_explainer_p,
                          protected = testing$SEX,
                          privileged = "male",
                          label = c("GBM_rs_uniform", "GBM_rs_pref"))

# visualize fairness check
plot(fobject)


####################################################################
################# REMOVING DISPARATE IMPACT ########################
####################################################################

# this script only works with dataframes with numeric variables

# keep only numeric variables in testing
testing_num <- testing[,sapply(testing, is.numeric)]
# cbind SEX as factor to testing_num
testing_num <- cbind(testing_num, SEX = testing$SEX)

# removing disparate impact
fixed_data <- fairmodels::disparate_impact_remover(
  data = testing_num,
  protected = testing_num$SEX,
  features_to_transform = "AGE",
  lambda = 1
)

# add back all factor variables
fixed_data <- cbind(fixed_data, KILLIP = testing$KILLIP)
fixed_data <- cbind(fixed_data, MILOC = testing$MILOC)
fixed_data <- cbind(fixed_data, PMI = testing$PMI)
fixed_data <- cbind(fixed_data, SMK = testing$SMK)
fixed_data <- cbind(fixed_data, TX = testing$TX)

# run GBM model on data where disparate impact is removed
set.seed(6363)
gbm_model     <- gbm(DAY30 ~. , data = fixed_data, distribution = "bernoulli")

# explain model
gbm_explainer_dir <- explain(gbm_model,
                             data = fixed_data[,-1],
                             y = testing$DAY30,
                             label = "gbm_disp_imp")

# model performance on data
model_performance(gbm_explainer_dir)

# fairness exploration
fobject <- fairness_check(fobject, gbm_explainer_dir,
                          protected = testing$SEX, 
                          privileged = "male",
                          label = c("GBM_disp_imp"))
plot(fobject)



####################################################################
####################################################################
####################### POST-PROCESSING ############################
####################################################################
####################################################################

####################################################################
######################## ROC-PIVOT #################################
####################################################################

set.seed(1)
gbm_model <-gbm(salary ~. , data = adult, distribution = "bernoulli")
gbm_explainer <- explain(gbm_model,
                         data = adult[,-1],
                         y = adult$salary,
                         verbose = FALSE)

gbm_explainer_r <- roc_pivot(gbm_explainer,
                             protected = protected,
                             privileged = "Male")


fobject <- fairness_check(fobject, gbm_explainer_r, 
                          label = "gbm_roc",  # label as vector for explainers
                          verbose = FALSE) 

plot(fobject)

####################################################################
################### CUTOFF MANIPULATION ############################
####################################################################

set.seed(1)
gbm_model <-gbm(salary ~. , data = adult, distribution = "bernoulli")
gbm_explainer <- explain(gbm_model,
                         data = adult[,-1],
                         y = adult$salary,
                         verbose = FALSE)

# test fairness object
fobject_test <- fairness_check(gbm_explainer, 
                               protected = protected, 
                               privileged = "Male",
                               verbose = FALSE) 

plot(ceteris_paribus_cutoff(fobject_test, subgroup = "Female"))

fc <- fairness_check(gbm_explainer, fobject,
                     label = "gbm_cutoff",
                     cutoff = list(Female = 0.25),
                     verbose = FALSE)

fc$parity_loss_metric_data
plot(fc)

print(fc , colorize = FALSE)


####################################################################
####################################################################
########################## TRADE-OFF ###############################
############# FAIRNESS AND PREDICTIVE VALIDITY #####################
####################################################################

paf <- performance_and_fairness(fc, fairness_metric = "STP",
                                performance_metric = "accuracy")

plot(paf)




# install Python environment to use aif360 (it takes ~5-10 minutes)
install.packages("reticulate")
reticulate::install_miniconda()
reticulate::conda_list()
reticulate::py_config()

reticulate::conda_create(envname = "r-test")
reticulate::use_miniconda(condaenv = "r-test", required = TRUE)
aif360::install_aif360(envname = "r-test")
reticulate::use_miniconda(condaenv = "r-test", required = TRUE)

library(aif360)
load_aif360_lib()


