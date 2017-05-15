#!/usr/bin/env Rscript
#
#This script is to train a gradient boosting machine(tree) model at the first level  
# to classify whether the product is ordered or not.
#
#Inputs: 
#     train_set(1<=day<=77): file from the path 'data/processed/train_set.feather'
#     pesudo_test_set(78<=day<=92): file from the path 'data/processed/validation_set.feather'
#Outputs: 
#     gbm-*modelid*-*auc*: save the h2o models in the folder 'models/'
#     gbm-*modelid*-1stLevelPred.csv: the prediction on the untouched test set


library(data.table)
library(feather)
library(h2o)
#library(h2oEnsemble)

h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "20G")  #max mem size is the maximum memory to allocate to H2O
h2o.removeAll()

####################################################################
### Set-up the validation scheme                                 ###
####################################################################

train63d <- read_feather("../data/processed/end63_train.feather")
valid63d <- read_feather("../data/processed/end63_test.feather")

# define predictors
features <- fread("../data/feature_list.csv")
#treat day_mod_ features as categorical
features[str_detect(name,'day_mod_'),type := "categorical"]
#should not include them in the modeling
NOT_USE <- c("pid", "fold", "lineID", "deduplicated_pid")
#not useful features list
LESS_IMPORTANT_VARS <- c("category_is_na","campaignIndex_is_na",
                         "pharmForm_is_na", "content_part1", 
                         "content_part2", "content_part3", 
                         "total_units", "price_discount_p25",
                         "price_discount_p75")

cat_vars <-  setdiff(features[type == "categorical", name], c(NOT_USE, LESS_IMPORTANT_VARS))
cont_vars <- setdiff(features[type == "numeric", name], c(cat_vars, LESS_IMPORTANT_VARS))

#probably want to replace these features
HIGH_DIMENSION_VARS <- c("group", "content", "manufacturer", 
                         "category", "pharmForm")
REPLACE_HIGH_DIMENSION_VARS <- FALSE
if (REPLACE_HIGH_DIMENSION_VARS == TRUE){
  cat_vars <- setdiff(cat_vars, HIGH_DIMENSION_VARS)
}

label <- c("order", "order_qty")
all_preds <- c(cat_vars, cont_vars)
all_vars <- c(all_preds, label)

train_set.hex <- as.h2o(train63d[all_vars])
validation_set.hex <- as.h2o(valid63d[all_vars])
rm(train63d, valid63d)

# factorize the categorical variables
for (c in cat_vars) {
  train_set.hex[c] <- as.factor(train_set.hex[c])
}

for (c in cat_vars) {
  validation_set.hex[c] <- as.factor(validation_set.hex[c])
}

####################################################################
### modeling part - Grid Search                                 ###
####################################################################

# GBM hyperparamters
gbm_params <- list( max_depth = seq(5, 13, 1),
                    sample_rate = seq(0.5, 1.0, 0.1),
                    min_rows = c(2,4,6),
                    col_sample_rate = seq(0.5, 1.0, 0.1),
                    ## search a large space of column sampling rates per tree
                    col_sample_rate_per_tree = seq(0.5, 1, 0.1), 
                    ## search a few minimum required relative error improvement thresholds for a split to happen
                    min_split_improvement = c(0, 1e-8, 1e-6, 1e-4),
                    ## try all histogram types (QuantilesGlobal and RoundRobin are good for numeric columns with outliers)
                    histogram_type = c("UniformAdaptive","QuantilesGlobal","RoundRobin"))

# Random Grid Search
# Ref: https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/product/tutorials/gbm/gbmTuning.Rmd
search_criteria <- list(strategy = "RandomDiscrete", 
                         # train no more than 6 models
                         max_models = 6,
                         ## random number generator seed to make sampling of parameter combinations reproducible
                         seed = 1234,                        
                         ## early stopping once the leaderboard of the top 5 models is 
                         #converged to 0.1% relative difference
                         stopping_rounds = 5,                
                         stopping_metric = "AUC",
                         stopping_tolerance = 1e-3)

# Train and validate a grid of GBMs for parameter tuning
gbm_grid <- h2o.grid(algorithm = "gbm",
                      hyper_params = gbm_params,
                      search_criteria = search_criteria,
                      x = all_preds, 
                      y = "order",
                      grid_id = "gbm_grid",
                      training_frame = train_set.hex,
                      validation_frame = validation_set.hex,
                      ntrees = 1000,
                      learn_rate = 0.05,
                      learn_rate_annealing = 0.99,
                      weights_column = "order_qty",
                      ## early stopping once the validation AUC doesn't improve 
                      #by at least 0.01% for 5 consecutive scoring events
                      stopping_rounds = 5, 
                      stopping_tolerance = 1e-4,
                      stopping_metric = "AUC", 
                      seed = 1234
                      )

sorted_GBM_Grid <- h2o.getGrid(grid_id = "gbm_grid", 
                               sort_by = "auc", 
                              decreasing = TRUE)
print(sorted_GBM_Grid)
#gbm_models <- lapply(gbm_grid@model_ids, function(model_id) h2o.getModel(model_id))

# remove the data in h2o
h2o.rm(train_set.hex)
h2o.rm(validation_set.hex)

####################################################################
### Retain the model on train77d                                 ###
####################################################################
#Load train77d and test77d dataset
train77d <- read_feather("../data/processed/end77_train.feather")
test77d <- read_feather("../data/processed/end77_test.feather")

train77d_index_df <- train77d[c("lineID")]
test77d_index_df <- test77d[c("lineID")]

#Load into the h2o environment
retrain_set.hex <- as.h2o(train77d[all_vars])
test_set.hex <- as.h2o(test77d[all_vars])

# factorize the categorical variables
for(c in cat_vars){
  retrain_set.hex[c] <- as.factor(retrain_set.hex[c])
}

for(c in cat_vars){
  test_set.hex[c] <- as.factor(test_set.hex[c])
}

rm(train77d, test77d)

# Only choose the top 3 models and persist the retrained model
# Note: need to refit model including the pesudo validation set
for (i in 1:3) {
  gbm <- h2o.getModel(sorted_GBM_Grid@model_ids[[i]])
  retrained_gbm <- do.call(h2o.gbm,
                   ## update parameters in place
                   {
                     p <- gbm@parameters        # the same seed
                     p$model_id = NULL          ## do not overwrite the original grid model
                     p$training_frame = train_set.hex   ## use the full training dataset
                     p$validation_frame = NULL  ## no validation frame
                     p
                   }
  )
  print(gbm@model_id)
  ## Get the AUC on the hold-out test set
  retrained_gbm_auc <- round(h2o.auc(h2o.performance(retrained_gbm, newdata = test_set.hex)),4)
  preds_train77d <- as.data.frame(h2o.predict(retrained_gbm, retrain_set.hex))[,3]
  preds_test77d <- as.data.frame(h2o.predict(retrained_gbm, test_set.hex))[,3]
  preds_train77d <- cbind(train77d_index_df, preds_train77d)
  preds_test77d <- cbind(test77d_index_df, preds_test77d)
  newnames = paste("gbm",i,sep="")
  names(preds_train77d)[2] = newnames
  names(preds_test77d)[2] = newnames
  
  # save the retrained model to regenerate the predictions for 2nd level modeling 
  # and possibly useful for ensemble
  h2o.saveModel(retrained_gbm, paste("../models/1stLevel/h2o_gbm",retrained_gbm_auc,sep = '-'), force = TRUE)
  write_feather(preds_train77d, paste0("../data/preds1stLevel/h2o_glm_train77d-",retrained_gbm_auc,'.feather'))
  write_feather(preds_test77d, paste0("../data/preds1stLevel/h2o_glm_test77d-",retrained_gbm_auc,'.feather'))
}

h2o.shutdown(prompt = FALSE)

