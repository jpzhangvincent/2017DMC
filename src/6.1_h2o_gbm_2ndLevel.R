#!/usr/bin/env Rscript
#
#This script is to train a gradient boosting machine(tree) model at the first level  
# to classify whether the product is ordered or not.


library(data.table)
library(feather)
library(stringr)
library(h2o)
#library(h2oEnsemble)

h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "20G")  #max mem size is the maximum memory to allocate to H2O
h2o.removeAll()

####################################################################
### Set-up the validation scheme                                 ###
####################################################################

train63d <- read_feather("../data/processed/end63_train_2nd.feather")
valid63d <- read_feather("../data/processed/end63_test_2nd.feather")

# define predictors from original features
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
REPLACE_HIGH_DIMENSION_VARS <- TRUE
if (REPLACE_HIGH_DIMENSION_VARS == TRUE){
  cat_vars <- setdiff(cat_vars, HIGH_DIMENSION_VARS)
}

#1st level predictions features(meta-model features)
column_names <- names(train63d)
model_features <- column_names[str_detect(column_names, 'preds_')]

label <- c("revenue")
# combine all features
all_preds <- c(cat_vars, cont_vars, model_features)
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
                    quantile_alpha = seq(0.2, 0.8, 0.1),
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
                        max_models = 8,
                        ## random number generator seed to make sampling of parameter combinations reproducible
                        seed = 1234,                        
                        ## early stopping once the leaderboard of the top 5 models is 
                        #converged to 0.1% relative difference
                        stopping_rounds = 5,                
                        stopping_metric = "RMSE",
                        stopping_tolerance = 1e-3)

# Train and validate a grid of GBMs for parameter tuning
gbm_grid <- h2o.grid(algorithm = "gbm",
                     hyper_params = gbm_params,
                     search_criteria = search_criteria,
                     x = all_preds, 
                     y = label,
                     distribution = "quantile",
                     grid_id = "gbm_grid",
                     training_frame = train_set.hex,
                     validation_frame = validation_set.hex,
                     ntrees = 1000,
                     learn_rate = 0.05,
                     learn_rate_annealing = 0.99,
                     ## early stopping once the validation AUC doesn't improve 
                     #by at least 0.01% for 5 consecutive scoring events
)

sorted_GBM_Grid <- h2o.getGrid(grid_id = "gbm_grid", 
                               sort_by = "rmse", 
                               decreasing = TRUE)
print(sorted_GBM_Grid)
#gbm_models <- lapply(gbm_grid@model_ids, function(model_id) h2o.getModel(model_id))
#save model

# remove the data in h2o
h2o.rm(train_set.hex)
h2o.rm(validation_set.hex)

####################################################################
### Retain the model on train77d                                 ###
####################################################################
#Load train77d and test77d dataset
train77d <- read_feather("../data/processed/end77_train_2nd.feather")
test77d <- read_feather("../data/processed/end77_test_2nd.feather")

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

# Only choose the top 4 models and persist the retrained model
# Note: need to refit model including the pesudo validation set
for (i in 1:4) {
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
  retrained_gbm_rmse <- round(h2o.rmse(h2o.performance(retrained_gbm, newdata = test_set.hex)),4)
  preds_train77d <- as.data.frame(h2o.predict(retrained_gbm, retrain_set.hex))[,3]
  preds_test77d <- as.data.frame(h2o.predict(retrained_gbm, test_set.hex))[,3]
  preds_train77d <- cbind(train77d_index_df, preds_train77d)
  preds_test77d <- cbind(test77d_index_df, preds_test77d)
  newnames = paste("preds_gbm",i,sep ="")
  names(preds_train77d)[2] = newnames
  names(preds_test77d)[2] = newnames
  
  # save the retrained model to regenerate the predictions for 2nd level modeling 
  # and possibly useful for ensemble
  #h2o.saveModel(retrained_gbm, paste("../models/2ndLevel/h2o_gbm",retrained_gbm_rmse,sep = '-'), force = TRUE)
  #write_feather(preds_train77d, paste0("../data/preds2ndLevel/end77d_train_gbm_",retrained_gbm_rmse,'.feather'))
  # train a third level ensemble model
  write_feather(preds_test77d, paste0("../data/preds2ndLevel/end77d_test_gbm_",retrained_gbm_rmse,'.feather'))
}

####################################################################
### Retain the model on train92d                                 ###
####################################################################
#Load train92d and test92d dataset
train92d <- read_feather("../data/processed/end92_train_2nd.feather")
test92d <- read_feather("../data/processed/end92_test_2nd.feather")

train92d_index_df <- train92d[c("lineID")]
test92d_index_df <- test92d[c("lineID")]

#Load into the h2o environment
retrain_set.hex <- as.h2o(train92d[all_vars])
test_set.hex <- as.h2o(test92d[all_preds])

# factorize the categorical variables
for(c in cat_vars){
  retrain_set.hex[c] <- as.factor(retrain_set.hex[c])
}

for(c in cat_vars){
  test_set.hex[c] <- as.factor(test_set.hex[c])
}

rm(train92d, test92d)

# Only choose the top 4 models and persist the retrained model
# Note: need to refit model including the pesudo validation set
for (i in 1:4) {
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
  retrained_gbm_rmse <- round(h2o.rmse(h2o.performance(retrained_gbm, newdata = test_set.hex)),4)
  preds_train92d <- as.data.frame(h2o.predict(retrained_gbm, retrain_set.hex))[,3]
  preds_test92d <- as.data.frame(h2o.predict(retrained_gbm, test_set.hex))[,3]
  preds_train92d <- cbind(train92d_index_df, preds_train92d)
  preds_test92d <- cbind(test92d_index_df, preds_test92d)
  newnames = paste("preds_gbm",i,sep ="")
  names(preds_train92d)[2] = newnames
  names(preds_test92d)[2] = newnames
  
  # save the retrained model to regenerate the predictions for 2nd level modeling 
  # and possibly useful for ensemble
  #h2o.saveModel(retrained_gbm, paste("../models/2ndLevel/h2o_gbm",retrained_gbm_rmse,sep = '-'), force = TRUE)
  #write_feather(preds_train92d, paste0("../data/preds2ndLevel/end92d_train_gbm_",retrained_gbm_rmse,'.feather'))
  write_feather(preds_test92d, paste0("../data/preds2ndLevel/end92_test_gbm_",retrained_gbm_rmse,'.feather'))
}

h2o.shutdown(prompt = FALSE)

