#!/usr/bin/env Rscript
#
# H2O GLM
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
glm_params <- list( alpha = c(0, 1e-4, 1e-2, 0.1,0.15,0.2,0.25, 0.5, 0.7,0.8,0.9, 1),
                    lambda = c(1e-4, 1e-2,1e-3, 0.1, 0.15,0.2, 0.25, 0.7,0.8,0.9,1))

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

