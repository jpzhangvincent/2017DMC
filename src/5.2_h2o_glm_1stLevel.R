#!/usr/bin/env Rscript
#
# H2O GLM
library(feather)
library(h2o)
library(data.table)
library(stringr)
#library(h2oEnsemble)
h2o.init(nthreads = 36, #Number of threads -1 means use all cores on your machine
         max_mem_size = "20G")  #max mem size is the maximum memory to allocate to H2O
h2o.removeAll()

####################################################################
### Set-up the validation scheme                                 ###
####################################################################

train63d <- read_feather("../data/processed/end63_train.feather")
valid63d <- read_feather("../data/processed/end63_test.feather")

train63d_index_df <- train63d[c("lineID")]
valid63d_index_df <- valid63d[c("lineID")]
# define predictors
features <- fread("../data/processed/feature_list.csv")
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

label <- c("order")
all_preds <- c(cat_vars, cont_vars)
all_vars <- c(all_preds, label)

train_set.hex<-as.h2o(train63d[all_vars])
validation_set.hex<-as.h2o(valid63d[all_vars])
rm(train63d, valid63d)

# factorize the categorical variables
for(c in cat_vars){
  train_set.hex[c] <- as.factor(train_set.hex[c])
}

for(c in cat_vars){
  validation_set.hex[c] <- as.factor(validation_set.hex[c])
}

####################################################################
### modeling part - Grid Search                                 ###
####################################################################

# GLM hyperparamters
# alpha_opts = list(list(.0001), list(.00001),list(.000001))
# lambda_opts = list(list(.0001), list(.00001),list(.000001))
# glm_params = list(alpha = alpha_opts,lambda = lambda_opts)
glm_params <- list( alpha = c(1e-2, 0.1,0.15,0.2,0.25,0.3,0.35,0.8,0.85,0.9),
                    lambda = c(1e-4, 1e-2,1e-3, 0.1, 0.15,0.2,0.25,0.8,0.85,0.9,0.95))

# Random Grid Search
search_criteria <- list(strategy = "RandomDiscrete", 
                         # train no more than 10 models
                         max_models = 8,
                         ## random number generator seed to make sampling of parameter combinations reproducible
                         seed = 1234,                        
                         ## early stopping once the leaderboard of the top 5 models is 
                         #converged to 0.1% relative difference
                         stopping_rounds = 5,                
                         stopping_metric = "AUC",
                         stopping_tolerance = 1e-3)

# Train and validate a grid of glms for parameter tuning
glm_grid <- h2o.grid(algorithm = "glm",
                     family = "binomial",
                     hyper_params = glm_params,
                     search_criteria = search_criteria,
                     x = all_preds, 
                     y = "order",
                     grid_id = "glm_grid",
                     training_frame = train_set.hex,
                     validation_frame = validation_set.hex)

sorted_GLM_Grid <- h2o.getGrid(grid_id = "glm_grid", 
                               sort_by = "auc", 
                               decreasing = TRUE)
print(sorted_GLM_Grid)

#save the top 3 models and generate the prediction features on 1-63d and 64-77d
for (i in 1:3){
  glm <- h2o.getModel(sorted_GLM_Grid@model_ids[[i]])
  h2o.saveModel(glm, paste("../models/1stLevel/h2o_glm",i), force=TRUE)
  preds_train63d <- as.data.frame(h2o.predict(glm, train_set.hex))[,3]
  preds_test63d <- as.data.frame(h2o.predict(glm, validation_set.hex))[,3]
  preds_train63d <- cbind(train63d_index_df, preds_train63d)
  preds_valid63d <- cbind(valid63d_index_df, preds_test63d)
  newnames = paste("preds_glm",i,sep="")
  names(preds_train63d)[2] = newnames
  names(preds_valid63d)[2] = newnames
  
  write_feather(preds_train63d, paste0("../data/preds1stLevel/end63d_train_glm",i,'.feather'))
  write_feather(preds_valid63d, paste0("../data/preds1stLevel/end63d_test_glm",i,'.feather'))
}

# remove the data in h2o
h2o.rm(train_set.hex)
h2o.rm(validation_set.hex)

####################################################################
### Retain the model on train77d  - ensemble learning            ###
####################################################################
#Load train77d and test77d dataset
train77d <- read_feather("../data/processed/end77_train.feather")
test77d <- read_feather("../data/processed/end77_test.feather")

train77d_index_df <- train77d[c("lineID")]
test77d_index_df <- test77d[c("lineID")]

#Load into the h2o environment
retrain_set.hex <-as.h2o(train77d[all_vars])
test_set.hex <-as.h2o(test77d[all_vars])

# factorize the categorical variables
for(c in cat_vars){
  retrain_set.hex[c] <- as.factor(retrain_set.hex[c])
}

for(c in cat_vars){
  test_set.hex[c] <- as.factor(test_set.hex[c])
}

#rm(train77d, test77d)

# Choose the top 3 models
# Note: need to refit model including the pesudo validation set
for (i in 1:3) {
  glm <- h2o.getModel(sorted_GLM_Grid@model_ids[[i]])
  retrained_glm <- do.call(h2o.glm,
                           ## update parameters in place
                           {
                             p <- glm@parameters  # the same seed
                             p$weights_column = NULL
                             p$model_id = NULL          ## do not overwrite the original grid model
                             p$training_frame = retrain_set.hex   ## use the full training dataset
                             p$validation_frame = NULL  ## no validation frame
                             p
                           }
  )
  print(glm@model_id)
  ## Get the AUC on the hold-out test set
  retrained_glm_auc <- round(h2o.auc(h2o.performance(retrained_glm, newdata = test_set.hex)),4)
  preds_train77d <- as.data.frame(h2o.predict(retrained_glm, retrain_set.hex))[,3]
  preds_test77d <- as.data.frame(h2o.predict(retrained_glm, test_set.hex))[,3]
  preds_train77d <- cbind(train77d_index_df, preds_train77d)
  preds_test77d <- cbind(test77d_index_df, preds_test77d)
  newnames = paste("preds_glm",i,sep="")
  names(preds_train77d)[2] = newnames
  names(preds_test77d)[2] = newnames
  
  # save the retrain model to regenerate the predictions for 2nd level modeling 
  # and possibly useful for ensemble
  #h2o.saveModel(retrained_glm, paste("../models/1stLevel/h2o_glm",retrained_glm_auc,sep='-'), force=TRUE)
  write_feather(preds_train77d, paste0("../data/preds1stLevel/end77d_train_glm",i,'.feather'))
  write_feather(preds_test77d, paste0("../data/preds1stLevel/end77d_test_glm",i,'.feather'))
}

####################################################################
### Retain the model on train92d                                 ###
####################################################################
#Load train92d and test92d dataset
train92d <- read_feather("../data/processed/end92_train.feather")
test92d <- read_feather("../data/processed/end92_test.feather")
test92d = test92d[,-84]

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

glm <- h2o.loadModel("../models/1stLevel/h2o_glm 1/glm_grid_model_0")
glm <- h2o.loadModel("../models/1stLevel/h2o_glm 2/glm_grid_model_5")
glm <- h2o.loadModel("../models/1stLevel/h2o_glm 3/glm_grid_model_2")


# Only choose the top 3 models and persist the retrained model
# Note: need to refit model including the pesudo validation set
for (i in 1:3) {
  glm <- h2o.getModel(sorted_GLM_Grid@model_ids[[i]])
  retrained_glm <- do.call(h2o.glm,
                           ## update parameters in place
                           {
                             p <- glm@parameters        # the same seed
                             p$model_id = NULL          ## do not overwrite the original grid model
                             p$training_frame = retrain_set.hex   ## use the full training dataset
                             p$validation_frame = NULL  ## no validation frame
                             p
                           }
  )
  print(glm@model_id)
  ## Get the AUC on the hold-out test set
  #retrained_glm_auc <- round(h2o.auc(h2o.performance(retrained_glm, newdata = test_set.hex)),4)
  #print(paste0("The AUC on 77-92 days: ", retrained_glm_auc))
  preds_train92d <- as.data.frame(h2o.predict(retrained_glm, retrain_set.hex))[,3]
  preds_test92d <- as.data.frame(h2o.predict(retrained_glm, test_set.hex))[,3]
  preds_train92d <- cbind(train92d_index_df, preds_train92d)
  preds_test92d <- cbind(test92d_index_df, preds_test92d)
  newnames = paste("preds_glm",i,sep="")
  names(preds_train92d)[2] = newnames
  names(preds_test92d)[2] = newnames
  
  # save the retrained model to regenerate the predictions for 2nd level modeling 
  # and possibly useful for ensemble
  #h2o.saveModel(retrained_glm, paste("../models/1stLevel/end92d_h2o_glm",retrained_glm_auc,sep = '-'), force = TRUE)
  write_feather(preds_train92d, paste0("../data/preds1stLevel/end92_train_glm",i,'.feather'))
  write_feather(preds_test92d, paste0("../data/preds1stLevel/end92_test_glm",i,'.feather'))
}

h2o.shutdown(prompt = FALSE)

