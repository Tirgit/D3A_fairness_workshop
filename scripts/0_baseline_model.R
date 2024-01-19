## Description: This script is used to create a baseline model for the health data.
##
## Author: Adrian G. Zucco and Tibor V. Varga
## Date Created: 2024-01-18
## Notes:
## Inspiration from https://www.kaggle.com/code/athosdamiani/lightgbm-with-tidymodels

## ------------------------------------
# Install list of libraries


library(skimr)
library(tidyverse)
library(tidymodels)
library(rsample)
library(lightgbm)
library(parsnip)
library(bonsai)
library(tune)
library(yardstick)

# Seed for the baseline model
seednr <- 02022024
set.seed(seednr)

# Load data
rawdata <- read_csv("../data/health_data.csv")

# Check data
skim(rawdata)

########### Pre-processing ################

data <- rawdata %>%
  # change format of day30 as factor
  mutate(DAY30 = as.factor(DAY30))


# Split data into train and test
split_data <- initial_split(data, prop = 0.70)
train <- training(split_data)
test <- testing(split_data)


# Data processing recipe
modelrecipe <- recipe(DAY30 ~ ., data = train) %>%
  # combine low frequency factor levels
  recipes::step_other(all_nominal(), threshold = 0.01) %>%
  # remove no variance predictors which provide no predictive information
  recipes::step_nzv(all_nominal(), -all_outcomes()) %>%
  prep()

# Have a look
skim(juice(prep(modelrecipe)))

# Define model
lightgbmmodel <- boost_tree(
  trees = 1000,
  learn_rate = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = 0.75,
  mtry = 5
) %>%
  set_engine("lightgbm", objective = "binary", metric = "binary_logloss") %>%
  set_mode("classification")



# Fit model
healthworkflow <- workflow() %>%
  add_recipe(modelrecipe) %>%
  # This model can be changed to any other model
  add_model(lightgbmmodel)

# Cross-validation
resamples <- vfold_cv(train, v = 3)

### Define hyperparameter optimization
training_grid <- parameters(lightgbmmodel) %>%
  finalize(train) %>%
  grid_random(size = 20)
head(training_grid)


# grid search, it takes some minutes
tune_grid <- healthworkflow %>%
  tune_grid(
    resamples = resamples,
    grid = training_grid,
    control = control_grid(verbose = TRUE),
    metrics = metric_set(roc_auc, pr_auc, accuracy, brier_class)
  )

# Show best models
show_best(tune_grid, "pr_auc", n = 10)

best_params <- select_best(tune_grid, "brier_class")
best_params

# Finalize workflow
healthworkflow <- healthworkflow %>%
  finalize_workflow(best_params)

# Fit model on test set
last_fit <- last_fit(healthworkflow,
  split = split_data
)

test_preds <- collect_predictions(last_fit)
test_preds

################# Global metrics #####################

# Calculate ROC AUC, accuracy, F1, PR AUC, Brier score
roc_auc(test_preds, truth = DAY30, .pred_1)
pr_auc(test_preds, truth = DAY30, .pred_1)
brier_class(test_preds, truth = DAY30, .pred_1)

accuracy(test_preds, truth = DAY30, .pred_class)
f_meas(test_preds, truth = DAY30, .pred_class)


# Plot ROC curve
roc_curve <- roc_curve(test_preds, truth = DAY30, .pred_1)
autoplot(roc_curve) +
  labs(title = "ROC curve") +
  theme_minimal()

# Plot PR curve
pr_curve <- pr_curve(test_preds, truth = DAY30, .pred_1)

autoplot(pr_curve) +
  labs(title = "PR curve") +
  theme_minimal()

# Confusion matrix
conf_mat <- conf_mat(test_preds, truth = DAY30, .pred_class)
conf_mat

