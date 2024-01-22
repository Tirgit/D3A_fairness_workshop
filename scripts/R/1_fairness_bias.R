## Description: This script is used to create a baseline GLM model for the health data.
##
## Author: Adrian G. Zucco and Tibor V. Varga
## Date Created: 2024-01-18
## Notes:
## Inspiration from: https://www.kaggle.com/code/athosdamiani/lightgbm-with-tidymodels
## Fairness tutorial: https://cran.r-project.org/web/packages/fairmodels/vignettes/Advanced_tutorial.html

## ------------------------------------
# Install list of libraries

# Install all the libraries with pak
# install.packages("pak")
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
                          epsilon = 0.8)

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
####################################################################
####################################################################

# https://towardsdatascience.com/fairmodels-lets-fight-with-biased-machine-learning-models-f7d66a2287fc
# https://cran.r-project.org/web/packages/fairmodels/vignettes/Advanced_tutorial.html

# load ADULT data
data("adult")
head(adult)
?adult

# predict salary - protected attibute: sex
adult$salary <- as.numeric(adult$salary) -1 # 0 if bad and 1 if good risk
protected <- adult$sex
adult <- adult[colnames(adult) != "sex"] # sex not specified

# making model
set.seed(1)
gbm_model <-gbm(salary ~. , data = adult, distribution = "bernoulli")

# making explainer object
gbm_explainer <- explain(gbm_model,
                         data = adult[,-1],
                         y = adult$salary,
                         colorize = FALSE)

# model performance on data
model_performance(gbm_explainer)

# fairness exploration
fobject <- fairness_check(gbm_explainer, 
                          protected  = protected, 
                          privileged = "Male", 
                          colorize = FALSE)

fobject
plot(fobject)

# PROBABILITIES
plot_density(fobject)

# METRIC DIFFERENCES
plot(metric_scores(fobject))

####################################################################
####################################################################
######################## PRE-PROCESSING ############################
####################################################################
####################################################################

####################################################################
################# REMOVING DISPARATE IMPACT ########################
####################################################################

# removing disparate impact
data_fixed <- disparate_impact_remover(data = adult, protected = protected, 
                                       features_to_transform = c("age", "hours_per_week",
                                                                 "capital_loss",
                                                                 "capital_gain"))

set.seed(1)
gbm_model     <- gbm(salary ~. , data = data_fixed, distribution = "bernoulli")
gbm_explainer_dir <- explain(gbm_model,
                             data = data_fixed[,-1],
                             y = adult$salary,
                             label = "gbm_dir",
                             verbose = FALSE)

# model performance on data
model_performance(gbm_explainer_dir)
#model_performance(gbm_explainer)

# fairness exploration
fobject <- fairness_check(gbm_explainer, gbm_explainer_dir,
                          protected = protected, 
                          privileged = "Male",
                          verbose = FALSE)
plot(fobject)

####################################################################
######################## REWEIGHTING ###############################
####################################################################

# calculation weights
weights <- reweight(protected = protected, y = adult$salary)

set.seed(1)
gbm_model <- gbm(salary ~. ,
                 data = adult,
                 weights = weights,
                 distribution = "bernoulli")

gbm_explainer_w <- explain(gbm_model,
                           data = adult[,-1],
                           y = adult$salary,
                           label = "gbm_weighted",
                           verbose = FALSE)

fobject <- fairness_check(fobject, gbm_explainer_w, verbose = FALSE)

plot(fobject)


####################################################################
######################### RESAMPLING ###############################
####################################################################

# to obtain probs we will use simple linear regression
probs <- glm(salary ~., data = adult, family = binomial())$fitted.values

uniform_indexes      <- resample(protected = protected,
                                 y = adult$salary)
preferential_indexes <- resample(protected = protected,
                                 y = adult$salary,
                                 type = "preferential",
                                 probs = probs)

set.seed(1)
gbm_model     <- gbm(salary ~. ,
                     data = adult[uniform_indexes,],
                     distribution = "bernoulli")

gbm_explainer_u <- explain(gbm_model,
                           data = adult[,-1],
                           y = adult$salary,
                           label = "gbm_uniform",
                           verbose = FALSE)

set.seed(1)
gbm_model     <- gbm(salary ~. ,
                     data = adult[preferential_indexes,],
                     distribution = "bernoulli")

gbm_explainer_p <- explain(gbm_model,
                           data = adult[,-1],
                           y = adult$salary,
                           label = "gbm_preferential",
                           verbose = FALSE)

fobject <- fairness_check(fobject, gbm_explainer_u, gbm_explainer_p, 
                          verbose = FALSE)
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






