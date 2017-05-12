#!/usr/bin/env Rscript
#
#This script is to train a gradient boosting machine(tree) model at the first level  
# to classify whether the product is ordered or not.
#
#Inputs: 
#     train_set(1<=day<=77): file from the path 'data/processed/train_set.feather'
#     pesudo_test_set(78<=day<=92): file from the path 'data/processed/validation_set.feather'
#Outputs: 
#     rf-*modelid*-*auc*: save the h2o models in the folder 'models/'
#     rf-*modelid*-1stLevelPred.csv: the prediction on the untouched test set


library(data.table)
library(feather)
library(h2o)
#library(h2oEnsemble)

h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "12G",
         enable_assertions = FALSE)  #max mem size is the maximum memory to allocate to H2O

setwd("C:/Users/Olivia/Desktop")

train_set <- read_feather("end63_train.feather") #day 1-63
pesudo_test_set <- read_feather("end63_test.feather") #day 64-77

train_set.hex <- as.h2o(train_set, destination_frame = "train_set.hex")
pesudo_test_set.hex <- as.h2o(pesudo_test_set, destination_frame = "test_set.hex")

# factorize the categorical variables
train_set.hex$order <- as.factor(train_set.hex$order)
train_set.hex$manufacturer <- as.factor(train_set.hex$manufacturer)
train_set.hex$pharmForm <- as.factor(train_set.hex$pharmForm)
train_set.hex$group <- as.factor(train_set.hex$group)
train_set.hex$unit <- as.factor(train_set.hex$unit)
train_set.hex$category <- as.factor(train_set.hex$category)
train_set.hex$campaignIndex <- as.factor(train_set.hex$campaignIndex)
train_set.hex$salesIndex <- as.factor(train_set.hex$salesIndex)
train_set.hex$adFlag <- as.factor(train_set.hex$adFlag)
  train_set.hex$last_adFlag <- as.factor(train_set.hex$last_adFlag)
train_set.hex$availability <- as.factor(train_set.hex$availability)
  train_set.hex$last_avaibility <- as.factor(train_set.hex$last_avaibility)
train_set.hex$group_beginNum <- as.factor(train_set.hex$group_beginNum)
train_set.hex$genericProduct <- as.factor(train_set.hex$genericProduct)
train_set.hex$content <- as.factor(train_set.hex$content)
  train_set.hex$avaibility_transition <- as.factor(train_set.hex$avaibility_transition)
  train_set.hex$adFlag_transition <- as.factor(train_set.hex$adFlag_transition)

pesudo_test_set.hex$order <- as.factor(pesudo_test_set.hex$order)
pesudo_test_set.hex$manufacturer <- as.factor(pesudo_test_set.hex$manufacturer)
pesudo_test_set.hex$pharmForm <- as.factor(pesudo_test_set.hex$pharmForm)
pesudo_test_set.hex$group <- as.factor(pesudo_test_set.hex$group)
pesudo_test_set.hex$unit <- as.factor(pesudo_test_set.hex$unit)
pesudo_test_set.hex$category <- as.factor(pesudo_test_set.hex$category)
pesudo_test_set.hex$campaignIndex <- as.factor(pesudo_test_set.hex$campaignIndex)
pesudo_test_set.hex$salesIndex <- as.factor(pesudo_test_set.hex$salesIndex)
pesudo_test_set.hex$adFlag <- as.factor(pesudo_test_set.hex$adFlag)
  pesudo_test_set.hex$last_adFlag <- as.factor(pesudo_test_set.hex$last_adFlag)
pesudo_test_set.hex$availability <- as.factor(pesudo_test_set.hex$availability)
  pesudo_test_set.hex$last_avaibility <- as.factor(pesudo_test_set.hex$last_avaibility)
  pesudo_test_set.hex$group_beginNum <- as.factor(pesudo_test_set.hex$group_beginNum)
pesudo_test_set.hex$genericProduct <- as.factor(pesudo_test_set.hex$genericProduct)
pesudo_test_set.hex$content <- as.factor(pesudo_test_set.hex$content)
  pesudo_test_set.hex$avaibility_transition <- as.factor(pesudo_test_set.hex$avaibility_transition)
  pesudo_test_set.hex$adFlag_transition <- as.factor(pesudo_test_set.hex$adFlag_transition)



# response and predictors
# Note: may need to update
response <- "order"
predictors <- setdiff(names(train_set), c("pid", "lineID", "day","order", "basket", "click", "revenue",
                                          "num_items_bought", "weight_qty", "fold_indicator", 
                                          "content_part1", "content_part2", "content_part3",'fold','order_qty'))
print(predictors)


# random forest hyperparamters
rf_params <- list( #ntrees = seq(50, 200, 20),
                    max_depth = seq(5, 13, 1),
                    sample_rate = seq(0.5, 1.0, 0.1),
                    #min_rows = c(2,4,6),
                   col_sample_rate_change_per_level = seq(0.5, 2.0, 0.2),
                    ## search a large space of column sampling rates per tree
                    col_sample_rate_per_tree = seq(0.5, 1, 0.1), 
                    ## search a few minimum required relative error improvement thresholds for a split to happen
                    min_split_improvement = c(0,1e-8,1e-6,1e-4),
                    ## try all histogram types (QuantilesGlobal and RoundRobin are good for numeric columns with outliers)
                    histogram_type = c("UniformAdaptive","QuantilesGlobal","RoundRobin"))
# Random Grid Search

search_criteria2 <- list(strategy = "RandomDiscrete", 
                         # train no more than 10 models
                         max_models = 10,
                         ## random number generator seed to make sampling of parameter combinations reproducible
                         seed = 1234,                        
                         ## early stopping once the leaderboard of the top 5 models is 
                         #converged to 0.1% relative difference
                         stopping_rounds = 5,                
                         stopping_metric = "AUC",
                         stopping_tolerance = 1e-3)

# Train and validate a grid of RFs for parameter tuning
rf_grid <- h2o.grid(algorithm = "randomForest",
                     hyper_params = rf_params,
                     search_criteria = search_criteria2,
                     x = predictors, 
                     y = response,
                     grid_id = "rf_grid1",
                     training_frame = train_set.hex,
                     validation_frame = pesudo_test_set.hex,
                     ntrees = 1000,
                     #weights_column = "weight_qty",
                     ## early stopping once the validation AUC doesn't improve 
                     #by at least 0.01% for 5 consecutive scoring events
                     stopping_rounds = 5, 
                     stopping_tolerance = 1e-4,
                     stopping_metric = "AUC", 
                     score_tree_interval = 10,
                     seed = 1234
)

rf_best <- h2o.randomForest(         ## h2o.randomForest function
  x = predictors,
  y = response,
  training_frame = train_set.hex,
  validation_frame = pesudo_test_set.hex,                          ## the target index (what we are predicting)
  model_id = "rf_covType_v1",    ## name the model in H2O
  ##   not required, but helps use Flow
  ntrees = 110,                  ## use a maximum of 200 trees to create the
  stopping_rounds = 5,  # changable
  stopping_metric = "AUC",  # changable
  stopping_tolerance = 1e-4,  # changable## Stop fitting new trees when the 2-tree
  ##  average is within 0.001 (default) of
  ##  the prior two 2-tree averages.
  ##  Can be thought of as a convergence setting
  score_each_iteration = T,      ## Predict against training and validation for
  ##  each tree. Default will skip several.
  max_depth = 12,
  sample_rate = 0.7,
  col_sample_rate_change_per_level = 1.9,
  ## search a large space of column sampling rates per tree
  col_sample_rate_per_tree = 0.8,
  ## search a few minimum required relative error improvement thresholds for a split to happen
  min_split_improvement = 1e-6,
  ## try all histogram types (QuantilesGlobal and RoundRobin are good for numeric columns with outliers)
  histogram_type = "RoundRobin"
)

sorted_RF_Grid <- h2o.getGrid(grid_id = "rf_grid1", 
                               sort_by = "auc", 
                               decreasing = TRUE)
print(sorted_RF_Grid)
#rf_models <- lapply(rf_grid@model_ids, function(model_id) h2o.getModel(model_id))
==========================================================================================================
  
  
  
  
# Only choose the top 5 models and persist the retrained model
# Note: need to refit model including the pesudo validation set
for (i in 1:5) {
  rf <- h2o.getModel(sorted_RF_Grid@model_ids[[i]])
  retrained_rf <- do.call(h2o.randomForest,
                           ## update parameters in place
                           {
                             p <- rf@parameters  # the same seed
                             p$model_id = NULL          ## do not overwrite the original grid model
                             p$training_frame = train_set.hex   ## use the full training dataset
                             p$validation_frame = NULL  ## no validation frame
                             p
                           }
  )
  print(rf@model_id)
  ## Get the AUC on the hold-out test set
  retrained_rf_auc <- h2o.auc(h2o.performance(retrained_rf, newdata = pesudo_test_set.hex))
  preds <- h2o.predict(retrained_rf, pesudo_test_set.hex)
  # save the retrain model to regenerate the predictions for 2nd level modeling 
  # and possibly useful for ensemble
  h2o.saveModel(retrained_rf, paste("data/models/rf",rf@model_id,round(retrained_rf_auc,3),sep='-'), force=TRUE)
  h2o.exportFile(preds, paste0("data/interim/rf",rf@model_id,'1stLevelPred.csv',sep='-'), force=TRUE)
}
