#!/usr/bin/env Rscript
#
# H2O GLM
library(feather)
library(h2o)
library(data.table)
library(stringr)
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
retrain_set.hex <-as.h2o(train77d[all_vars])
test_set.hex <-as.h2o(test77d[all_vars])

# factorize the categorical variables
for(c in cat_vars){
  retrain_set.hex[c] <- as.factor(retrain_set.hex[c])
}

for(c in cat_vars){
  test_set.hex[c] <- as.factor(test_set.hex[c])
}

rm(train77d, test77d)

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
  newnames = paste("glm",i,sep="")
  names(preds_train77d)[2] = newnames
  names(preds_test77d)[2] = newnames
  
  # save the retrain model to regenerate the predictions for 2nd level modeling 
  # and possibly useful for ensemble
  h2o.saveModel(retrained_glm, paste("../models/1stLevel/h2o_glm",retrained_glm_auc,sep='-'), force=TRUE)
  write_feather(preds_train77d, paste0("../data/preds1stLevel/h2o_glm_train77d-",retrained_glm_auc,'.feather'))
  write_feather(preds_test77d, paste0("../data/preds1stLevel/h2o_glm_test77d-",retrained_glm_auc,'.feather'))
}
h2o.shutdown(prompt = FALSE)

