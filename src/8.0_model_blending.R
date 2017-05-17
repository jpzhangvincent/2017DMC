#!/usr/bin/env Rscript

library(data.table)
library(feather)
library(h2o)

DIR <- "../data/layer3"
#list all the test77d files in the data/preds2ndLevel folder

path <- file.path(DIR, "end77_test_layer3.feather")
df <- data.table(read_feather(path))

#fit a glmnet on the true revenue(with cross validation?)
predictors <- setdiff(colnames(df), "revenue")

# grid over `tweedie_variance_power`
# select the values for `tweedie_variance_power` to grid over
hyper_params <- list(tweedie_variance_power = c(1.2, 1.4, 1.6, 1.8),
                     alpha = c(0.1, 0.3, 0.6, 0.8), 
                     lambda = c(1e-2, 0.1, 0.15,0.2,0.25,0.5,0.8))

# this example uses cartesian grid search because the search space is small
# and we want to see the performance of all models. For a larger search space use
# random grid search instead: {'strategy': "RandomDiscrete"}

# build grid search with previously selected hyperparameters
grid <- h2o.grid(x = predictors, y = "revenue", training_frame = df, nfold = 3,
                 family = 'tweedie', algorithm = "glm", grid_id = "auto_grid", 
                 hyper_params = hyper_params, 
                 search_criteria = list(strategy = "RandomDiscrete"))

# Sort the grid models by rmse
sortedGrid <- h2o.getGrid("auto_grid", sort_by = "rmse", decreasing = FALSE)

#save the weights(model parameters) in "models/final_blending_weights.rda"
for (i in 1:3){
  blending_glm <- h2o.getModel(sortedGrid@model_ids[[i]])
  h2o.saveModel(blending_glm, paste0("../model/final_blending_model", i))
}

