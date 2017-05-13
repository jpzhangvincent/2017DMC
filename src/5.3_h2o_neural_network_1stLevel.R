#install.packages("h2o",repos="http://cran.rstudio.org")
#install.packages("feather",repos="htpp://cran.rstudio.org")
#install.packages("data.table")
library(h2o)
library(data.table)
library(feather)
h2o.init(nthreads = -1, max_mem_size = "30G")
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
### modeling part on train66d                                   ###
####################################################################

#### train on all the features in train63d dataset
dl_all <- h2o.deeplearning(
  model_id="dl_model_all", 
  training_frame = train_set.hex, 
  validation_frame = validation_set.hex,   ## validation dataset: used for scoring and early stopping
  x = all_preds,
  y = label,
  seed = 2017,
  activation="Tanh",  ## default
  hidden=c(256, 128, 64, 32),       ## default: 2 hidden layers with 200 neurons each
  epochs=10000,
  stopping_rounds=3,
  stopping_metric="AUC",
  stopping_tolerance=0.001,
  l1=0.000010,
  l2=0.010000,
  #variable_importances=T    ## not enabled by default
)

sprintf("Model performance for the model train on all the features in train63d dataset")
h2o.performance(model = dl_all, valid = T)

#### train on the categorical variables in train63d dataset
dl_cat <- h2o.deeplearning(
  model_id="dl_model_cat", 
  training_frame = train_set.hex, 
  validation_frame = validation_set.hex,   ## validation dataset: used for scoring and early stopping
  x = cat_vars,
  y = label,
  seed = 2017,
  activation="Tanh",  ## default
  hidden=c(200, 100, 50),       ## default: 2 hidden layers with 200 neurons each
  epochs=10000,
  stopping_rounds=3,
  stopping_metric="auc",
  stopping_tolerance=0.001,
  l1=0.000010,
  l2=0.010000,
  #variable_importances=T    ## not enabled by default
)
sprintf("Model performance for the model train on only categorical features in train63d dataset")
h2o.performance(model = dl_cat, valid = T)

# remove the data in h2o
h2o.rm(train_set.hex)
h2o.rm(validation_set.hex)

####################################################################
### Retain the model on train77d                                 ###
####################################################################
#Load train77d and test77d dataset
train77d <- read_feather("../data/processed/end77_train.feather")
test77d <- read_feather("../data/processed/end77_test.feather")

train77d_index_df <- train77d[c("lineID", "deduplicated_pid", "day")]
test77d_index_df <- test77d[c("lineID", "deduplicated_pid", "day")]

#Load into the h2o environment
retrain_set.hex <-as.h2o(train77d[all_vars])
test_set.hex <-as.h2o(test77d[all_vars])
rm(train77d, test77d)

#retain on the train77d dataset
retrained_dl_all <- do.call(h2o.deeplearning,
                         ## update parameters in place
                         {
                           p <- dl_all@parameters  # the same seed
                           p$model_id = NULL          ## do not overwrite the original grid model
                           p$training_frame = retrain_set.hex   ## use the full training dataset
                           p$validation_frame = NULL  ## no validation frame
                           p
                         })

sprintf("Model performance for the model train on all the features in train77d dataset")
h2o.auc(h2o.performance(model = retrained_dl_all, newdata = test_set.hex))

retrained_dl_cat <- do.call(h2o.deeplearning,
                            ## update parameters in place
                            {
                              p <- dl_cat@parameters  # the same seed
                              p$model_id = NULL          ## do not overwrite the original grid model
                              p$training_frame = retrain_set.hex   ## use the full training dataset
                              p$validation_frame = NULL  ## no validation frame
                              p
                            })

sprintf("Model performance for the model train on categorical features in train77d dataset")
h2o.auc(h2o.performance(model = retrained_dl_cat, newdata = test_set.hex))

# save the models
h2o.saveModel(object = retrained_dl_all, path = "../models/1stLevel/h2o_dl_all", force=TRUE)
h2o.saveModel(object = retrained_dl_cat, path = "../models/1stLevel/h2o_dl_cat", force=TRUE)

# save the predictions for the second level modeling
pred_all_train77d <- as.data.frame(h2o.predict(retrained_dl_all, newdata = retrain_set.hex))
pred_all_test77d <- as.data.frame(h2o.predict(retrained_dl_all, newdata = test_set.hex))
pred_all_train77d <- cbind(train77d_index_df, pred_all_train77d)
pred_all_test77d <- cbind(test77d_index_df, pred_all_test77d)

write_feather(pred_all_train77d, "../data/preds1stLevel/dl_all_train77d.feather")
write_feather(pred_all_test77d, "../data/preds1stLevel/dl_all_test77d.feather")

pred_cat_train77d <- as.data.frame(h2o.predict(retrained_dl_cat, newdata = retrain_set.hex))
pred_cat_test77d <- as.data.frame(h2o.predict(retrained_dl_cat, newdata = test_set.hex))
pred_cat_train77d <- cbind(train77d_index_df, pred_cat_train77d)
pred_cat_test77d <- cbind(test77d_index_df, pred_cat_test77d)
write_feather(pred_cat_train77d, "../data/preds1stLevel/dl_cat_train77d.feather")
write_feather(pred_cat_test77d, "../data/preds1stLevel/dl_cat_test77d.feather")

h2o.shutdown(prompt = FALSE)


######################### grid search before #############################
# activation_opt <- c("Rectifier", "Maxout", "Tanh")
# l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)
# l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)
# hyper_params <- list(activation = activation_opt, 
#                      l1 = l1_opt, 
#                      l2 = l2_opt)
# search_criteria <- list(strategy = "RandomDiscrete", 
#                         max_runtime_secs = 600)
# 
# dl_grid <- h2o.grid("deeplearning", 
#                     x = all_preds, 
#                     y = label,
#                     grid_id = "dl_grid",
#                     training_frame = train,
#                     validation_frame = valid,
#                     seed = 1234,
#                     hidden = c(500,400,500,400,500),
#                     hyper_params = hyper_params,
#                     search_criteria = search_criteria)
# 
# h2o.saveModel(object=dl_grid, path=getwd(), force=TRUE)
# 
# ## sort the grid
# 
#  dl_gridperf <- h2o.getGrid(grid_id = "dl_grid", 
#                            sort_by = "accuracy", 
#                            decreasing = TRUE)
# # print(dl_gridperf)
# 
# ## pick the best setting
# 
# best_dl_model_id <- dl_gridperf@model_ids[[1]]
# best_dl <- h2o.getModel(best_dl_model_id)
# 
# h2o.performance(best_dl,newdata=test)