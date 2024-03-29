## Description: This script is used to create a LightGBM model with hyperparameter tuning for the health data.
##
## Authors: Adrian G. Zucco, Alexandros Katsiferis, Tibor V. Varga
## Date Created: 2024-01-18
## Notes:
## Inspiration from https://www.kaggle.com/code/athosdamiani/lightgbm-with-tidymodels

## ------------------------------------
# Install list of libraries

# Install all the libraries with pak
# install.packages("pak", repos = "https://r-lib.github.io/p/pak/devel/")
# pak::pkg_install(c("skimr", "tidyverse", "tidymodels", "rsample", "lightgbm",
# "parsnip", "bonsai", "tune", "yardstick", "DALEX", "DALEXtra", "fairmodels",
# "ggplot2", "pROC"))

# Load libraries
library(skimr)
library(tidyverse)
library(tidymodels)
library(rsample)
library(lightgbm)
library(parsnip)
library(bonsai)
library(tune)
library(yardstick)
library(DALEX)
library(DALEXtra)
library(fairmodels)
library(pROC)
library(ggplot2)

# Seed for the baseline model
seednr <- 02022024
set.seed(seednr)

# Load data
rawdata <- read_csv("../../data/health_data.csv")

########### Pre-processing ################

# convert DAY30 to factor
rawdata$DAY30 <- factor(rawdata$DAY30, levels = c(0, 1), labels = c("no", "yes"))

# convert all character variables to factors
data <- rawdata %>%
  mutate_if(is.character, as.factor)

# Check data
skim(data)

# Split data into train and test
split_data <- initial_split(data, prop = 0.70, strata = DAY30)
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

# You can define you own models, see:
# https://www.tidymodels.org/find/parsnip/
# your_own_model <- xxxxx


# Fit model
healthworkflow <- workflow() %>%
  add_recipe(modelrecipe) %>%
  # This model can be changed to any other model
  add_model(lightgbmmodel)

# Fit your own model:
# healthworkflow <- workflow() %>%
#  add_recipe(modelrecipe) %>%
#  add_model(your_own_model)

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
  split = split_data,
  metrics = metric_set(roc_auc, pr_auc, brier_class, f_meas, accuracy)
)

test_preds <- collect_predictions(last_fit)
test_preds

################# Global metrics #####################
# Confusion matrix
conf_mat <- conf_mat(test_preds, truth = DAY30, .pred_class)
conf_mat

collect_metrics(last_fit)

# Plot ROC curve
result.roc <- roc(test$DAY30, test_preds$.pred_no) 
ggroc(result.roc, alpha = 0.5, colour = "red", linetype = 1, size = 1) + 
  theme_minimal() + 
  ggtitle("ROC curve - logistic regression") + 
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), color="grey", linetype="dashed")


# Plot PR curve
pr_curve <- pr_curve(test_preds, truth = DAY30, .pred_yes)

autoplot(pr_curve) +
  labs(title = "PR curve") +
  theme_minimal()


################# Metrics per group #####################
# Merge predictions with test data
groups_wit_preds <- test_preds %>%
  bind_cols(test %>% select(-DAY30)) %>% 
  group_by(SEX)

# Confusion matrix
conf_mat <- conf_mat(groups_wit_preds, truth = DAY30, .pred_class)

conf_mat$SEX[1]
conf_mat$conf_mat[[1]]

conf_mat$SEX[2]
conf_mat$conf_mat[[2]]

# Calculate ROC AUC, accuracy, F1, PR AUC, Brier score
roc_auc(groups_wit_preds, truth = DAY30, .pred_no)
pr_auc(groups_wit_preds, truth = DAY30, .pred_yes)
brier_class(groups_wit_preds, truth = DAY30, .pred_yes)
accuracy(groups_wit_preds, truth = DAY30, .pred_class)
f_meas(groups_wit_preds, truth = DAY30, .pred_class)
