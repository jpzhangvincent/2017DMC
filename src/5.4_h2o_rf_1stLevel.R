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
         max_mem_size = "15G",
         enable_assertions = FALSE)  #max mem size is the maximum memory to allocate to H2O

h2o.removeAll()

####################################################################
### Set-up the validation scheme                                 ###
####################################################################

train63d <- read_feather("../data/processed/end63_train.feather")
valid63d <- read_feather("../data/processed/end63_test.feather")
# define predictors
cat_vars <- c('adFlag',
              'availability',
              'manufacturer',
              'group',
              'content',
              'unit',
              'pharmForm',
              'genericProduct',
              'salesIndex',
              'category',
              'campaignIndex',
              'group_begin_num',
              'day_mod_7',
              'day_mod_10',
              'day_mod_14',
              'day_mod_28',
              'day_mod_30',
              'is_lower_price',
              'is_discount',
              'isgreater_discount',
              'price_gt_prev5',
              'price_lt_next5',
              'prev_availability',
              'prev_adFlag',
              'availability_trans',
              'adFlag_trans')

cont_vars <- c('day',
               'price',
               'rrp',
               'rrp_per_unit',
               'percent_of_day',
               'competitorPrice_is_na',
               'competitorPrice_imputed',
               'price_per_unit',
               'competitorPrice_per_unit',
               'price_diff',
               'price_discount',
               'competitorPrice_discount',
               'price_discount_diff',
               'price_discount_min',
               'price_discount_p25',
               'price_discount_med',
               'price_discount_p75',
               'price_discount_max',
               'price_discount_mad',
               'content_d7cnt',
               'group_d7cnt',
               'manufacturer_d7cnt',
               'unit_d7cnt',
               'pharmForm_d7cnt',
               'category_d7cnt',
               'campaignIndex_d7cnt',
               'salesIndex_d7cnt',
               'inter_gcucd7_cnt',
               'inter_gcucd10_cnt',
               'inter_gcucd30_cnt',
               'inter_gcuca_cnt',
               'prev_price',
               'prev5_price_avg',
               'prev5_price_min',
               'prev5_price_max',
               'next_price',
               'next5_price_avg',
               'next5_price_min',
               'next5_price_max',
               'prev_price_pct_chg',
               'prev5_price_diff',
               'next_price_pct_chg',
               'next5_price_diff',
               'num_pid_click',
               'prob_pid_click',
               'num_pid_basket',
               'prob_pid_basket',
               'num_pid_order',
               'prob_pid_order',
               'order_qty_eq_1_prob',
               'order_qty_gt_1_prob',
               'num_cons_orders',
               'prob_cons_orders',
               'cnt_click_byday7',
               'cnt_basket_byday7',
               'cnt_order_byday7',
               'avg_price_click_info',
               'avg_price_diff_click_info',
               'avg_price_disc_diff_click_info',
               'avg_price_basket_info',
               'avg_price_diff_basket_info',
               'avg_price_disc_diff_basket_info',
               'avg_price_order_info',
               'avg_price_diff_order_info',
               'avg_price_disc_diff_order_info',
               'click_propensity',
               'basket_propensity',
               'order_propensity',
               'avg_revenue_by_group_7',
               'avg_revenue_by_group_10',
               'avg_revenue_by_group_30')

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

# random forest hyperparamters
rf_params <- list( max_depth = seq(5, 13, 1),
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

# Train and validate a grid of RFs for parameter tuning
rf_grid <- h2o.grid(algorithm = "randomForest",
                     hyper_params = rf_params,
                     search_criteria = search_criteria,
                     x = all_preds, 
                     y = "order",
                     grid_id = "rf_grid",
                     training_frame = train_set.hex,
                     validation_frame = validation_set.hex,
                     ntrees = 1000,
                     ## early stopping once the validation AUC doesn't improve 
                     #by at least 0.01% for 5 consecutive scoring events
                     stopping_rounds = 5, 
                     stopping_tolerance = 1e-4,
                     stopping_metric = "AUC", 
                     score_tree_interval = 10,
                     seed = 27)

sorted_RF_Grid <- h2o.getGrid(grid_id = "rf_grid", 
                               sort_by = "auc", 
                               decreasing = TRUE)
print(sorted_RF_Grid)
#rf_models <- lapply(rf_grid@model_ids, function(model_id) h2o.getModel(model_id))
  
####################################################################
### Retain the model on train77d                                 ###
####################################################################
#Load train77d and test77d dataset
train77d <- read_feather("../data/processed/end77_train.feather")
test77d <- read_feather("../data/processed/end77_test.feather")

train77d_index_df <- train77d[c("lineID", "deduplicated_pid", "day")]
test77d_index_df <- test77d[c("lineID", "deduplicated_pid", "day")]

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
  rf <- h2o.getModel(sorted_RF_Grid@model_ids[[i]])
  retrained_rf <- do.call(h2o.randomForest,
                           ## update parameters in place
                           {
                             p <- rf@parameters  # the same seed
                             p$model_id = NULL          ## do not overwrite the original grid model
                             p$training_frame = retrain_set.hex   ## use the full training dataset
                             p$validation_frame = NULL  ## no validation frame
                             p
                           }
  )
  print(rf@model_id)
  ## Get the AUC on the hold-out test set
  retrained_rf_auc <- round(h2o.auc(h2o.performance(retrained_rf, newdata = test_set.hex)),4)
  preds_train77d <- as.data.frame(h2o.predict(retrained_rf, retrain_set.hex))[,3]
  preds_test77d <- as.data.frame(h2o.predict(retrained_rf, test_set.hex))[,3]
  preds_train77d <- cbind(train77d_index_df, preds_train77d)
  preds_test77d <- cbind(test77d_index_df, preds_test77d)
  newnames = paste("rf",i,sep="")
  names(preds_train77d)[4] = newnames
  names(preds_test77d)[4] = newnames
  
  # save the retrained model to regenerate the predictions for 2nd level modeling 
  # and possibly useful for ensemble
  h2o.saveModel(retrained_rf, paste("../models/1stLevel/h2o_rf",retrained_rf_auc,sep = '-'), force = TRUE)
  write_feather(preds_train77d, paste0("../data/preds1stLevel/h2o_rf_train77d-",retrained_rf_auc,'.feather'))
  write_feather(preds_test77d, paste0("../data/preds1stLevel/h2o_rf_test77d-",retrained_rf_auc,'.feather'))
}

h2o.shutdown(prompt = FALSE)
