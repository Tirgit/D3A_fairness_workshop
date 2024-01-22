# DEMONSTRATION: FAIRNESS EXPLORATION

# you only need to install packages once
install.packages("caret")
install.packages("pROC")
install.packages("ggplot2")
install.packages("fairness")
install.packages("fairmodels")
install.packages("DALEX")
install.packages("ranger")
install.packages("gbm")
install.packages("nnet")

# you need to load libraries each time you start R
library(caret)
library(pROC)
library(ggplot2)
library(fairness) # both fairness packages have a compas dataset!
library(fairmodels) # both fairness packages have a compas dataset!
library(DALEX)
library(ranger)
library(gbm)
library(nnet)


####################################################################
####################################################################
##################### INITIAL PREDICTIONS ##########################
####################################################################
####################################################################

# loading data
rawdata <- read_csv("../../data/health_data.csv")

rawdata$DAY30 <- factor(rawdata$DAY30, levels = c(0, 1), labels = c("no", "yes"))

data <- rawdata %>%
  mutate_if(is.character, as.factor)

# DATA PARTITIONING
set.seed(23456)
inTraining <- createDataPartition(data$DAY30, p = .75, list = FALSE)
training <- data[ inTraining,]
testing  <- data[-inTraining,]

# FIT PREDICTION MODEL WITH ETHNICITY AND SEX
fitControl <- trainControl(classProbs = TRUE, 
                           summaryFunction = twoClassSummary)
# note that this is a full model including all variables
glm.model <- train(DAY30 ~ ., 
                   training,
                   method="glm",
                   trControl = fitControl,
                   metric="ROC")
glm.model

# predict on test set
result.predicted.prob <- predict(glm.model, testing, type="prob")

# Draw ROC curve
result.roc <- roc(testing$DAY30, result.predicted.prob$yes) 

ggroc(result.roc, alpha = 0.5, colour = "red", linetype = 1, size = 1) + 
  theme_minimal() + 
  ggtitle("ROC curve - logistic regression") + 
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), color="grey", linetype="dashed")

predictions <- predict(glm.model, newdata = testing)
conf_matrix <- confusionMatrix(predictions, testing$DAY30)
conf_matrix


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
explainer_lm <- explain(glm.model, data = testing[,-1], y = y_testing)
fobject <- fairness_check(explainer_lm,
                          protected = testing$SEX,
                          privileged = "male",
                          epsilon = 0.8)

# CHECK IF FAIRNESS CHECK PASSED FOR 5 SELECTED METRICS
fobject

# VISUALIZE FAIRNESS CHECK
# FIRST, RAW METRICS
plot(metric_scores(fobject))
# THEN, RELATIVE METRICS
plot(fobject)

# NUMERIC OUTPUT
fobject$groups_data$train.formula$ACC
fobject$groups_data$train.formula$TPR

# PROBABILITIES
plot_density(fobject)

# ADD ANOTHER ALGORITHM TO THE PIPELINE
rf_model     <- ranger(Risk ~., data = german, probability = TRUE)
explainer_rf <- explain(rf_model, data = testing[,-1], y = y_testing)
fobject      <- fairness_check(explainer_rf, fobject)

# VISUALIZE BOTH ALGORITHMS
# FIRST, RAW METRICS
plot(metric_scores(fobject))
# THEN, RELATIVE METRICS
plot(fobject)

# NUMERIC OUTPUT
fobject$groups_confusion_matrices
fobject$groups_data$train.formula$ACC
fobject$groups_data$train.formula$TPR


####################################################################
####################################################################
###################### FAIRMODELS R PACKAGE ########################
################## LARGE SCALE METRIC COMPARISON ###################
####################################################################
####################################################################

# load COMPAS data
compas <- fairmodels::compas
?compas

# making sure outcome is 0 / 1
two_yr_recidivism <- factor(compas$Two_yr_Recidivism, levels = c(1,0))
y_numeric <- as.numeric(two_yr_recidivism) -1
compas$Two_yr_Recidivism <- two_yr_recidivism

df <- compas
df$Two_yr_Recidivism <- as.numeric(two_yr_recidivism) -1


# FITTING THREE ALGORITHMS
set.seed(4328)
gbm_model <- gbm(Two_yr_Recidivism ~., data = df)

lm_model <- glm(Two_yr_Recidivism~.,
                data=compas,
                family=binomial(link="logit"))

rf_model <- ranger(Two_yr_Recidivism ~.,
                   data = compas,
                   probability = TRUE)

# EXPLAINER
explainer_lm  <- explain(lm_model, data = compas[,-1], y = y_numeric)
explainer_rf  <- explain(rf_model, data = compas[,-1], y = y_numeric)
explainer_gbm <- explain(gbm_model, data = compas[,-1], y = y_numeric)

# FAIRNESS GRAPHS
fobject <- fairness_check(explainer_lm, explainer_rf, explainer_gbm,
                          protected = compas$Ethnicity,
                          privileged = "Caucasian")

# FAIRNESS HEATMAP
plot(fairness_heatmap(fobject))

# FAIRNESS RADAR
plot(fairness_radar(fobject))

# CHOSEN METRIC
plot(group_metric(fobject))


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






