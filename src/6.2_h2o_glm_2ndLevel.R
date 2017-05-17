#!/usr/bin/env Rscript
# 
library(data.table)
library(feather)
library(stringr)
library(h2o)

h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "20G")  #max mem size is the maximum memory to allocate to H2O
h2o.removeAll()

####################################################################
### Set-up the validation scheme                                 ###
####################################################################

train63d <- read_feather("../data/processed/end63_train_2nd.feather")
valid63d <- read_feather("../data/processed/end63_test_2nd.feather")

offset_logrithm = function( data_set){
  # create another revenue column called log_revenue
  data_set$log_revenue = log(1e-5+ data_set$revenue) - log(1e-5)
  data_set$loo_mean_revenue_by_pid = abs(1e-5+data_set$loo_mean_revenue_by_pid) - log(1e-5)
  
  # all columns need to transfer to log-scale
  need_to_log=c("avg_price_basket_info",
                "avg_price_click_info",
                "avg_price_order_info",
                "avg_revenue_by_group_10",
                "avg_revenue_by_group_30",
                "avg_revenue_by_group_7",
                "competitorPrice_per_unit",
                "loo_mean_revenue_by_pid",
                "next_price",
                "next5_price_avg",
                "next5_price_max",
                "next5_price_min",
                "prev_price",
                "prev5_price_avg",
                "prev5_price_diff",
                "prev5_price_max",
                "prev5_price_min",
                "price",
                "price_per_unit",
                "rrp",
                "rrp_per_unit")
  
  data_set[,need_to_log] = log(1e-5+ data_set[,need_to_log]) - log(1e-5)
  
  return(data_set)
}

newtrain63d = offset_logrithm(train63d)
newvalid63d = offset_logrithm(valid63d)


# define predictors from original features
features <- fread("feature_list.csv")
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
model_features = model_features[1:3]

label <- c("log_revenue")
# combine all features
all_preds <- c(cat_vars, cont_vars, model_features)
all_vars <- c(all_preds, label)

train_set.hex <- as.h2o(newtrain63d[all_vars])
validation_set.hex <- as.h2o(newvalid63d[all_vars])
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

# GLM hyperparamters
glm_params <- list( alpha = c(0.1,0.5,0.9),
                    lambda = c(0.1,0.5,0.9))

# Random Grid Search
search_criteria <- list(strategy = "RandomDiscrete", 
                        # train no more than 10 models
                        max_models = 8,
                        ## random number generator seed to make sampling of parameter combinations reproducible
                        seed = 1234,                        
                        ## early stopping once the leaderboard of the top 5 models is 
                        #converged to 0.1% relative difference
                        stopping_rounds = 5,                
                        stopping_metric = "MSE",
                        stopping_tolerance = 1e-3)

# Train and validate a grid of glms for parameter tuning
# tuning parameter p = c(0,1,1.3,1.5,1.8,2,3,4)
glm_grid <- h2o.grid(algorithm = "glm",
                     family = "tweedie",
                     tweedie_variance_power = 1.9,
                     hyper_params = glm_params,
                     search_criteria = search_criteria,
                     x = all_preds, 
                     y = label,
                     grid_id = "glm_grid",
                     training_frame = train_set.hex,
                     validation_frame = validation_set.hex)

fit = h2o.glm(x = all_preds, 
              y = label,
              training_frame = train_set.hex,
              validation_frame = validation_set.hex,
              family = "tweedie",
              tweedie_variance_power = 1.9
              )

sorted_GLM_Grid <- h2o.getGrid(grid_id = "glm_grid", 
                               sort_by = "rmse")
                               #decreasing = TRUE)
print(sorted_GLM_Grid)
glm <- h2o.getModel(sorted_GLM_Grid@model_ids[[1]])
preds_test63d <- as.data.frame(h2o.predict(glm, validation_set.hex))
newpred = exp(preds_test63d[,1]+log(1e-5))-10^-5
sqrt(mean((newpred-valid63d$revenue)^2))

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
  h2o.saveModel(retrained_glm, paste("../models/1stLevel/h2o_glm",retrained_glm_auc,sep='-'), force=TRUE)
  write_feather(preds_train77d, paste0("../data/preds1stLevel/h2o_glm_train77d-",retrained_glm_auc,'.feather'))
  write_feather(preds_test77d, paste0("../data/preds1stLevel/h2o_glm_test77d-",retrained_glm_auc,'.feather'))
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

