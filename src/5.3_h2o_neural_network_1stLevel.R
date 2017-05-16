#install.packages("h2o",repos="http://cran.rstudio.org")
#install.packages("feather",repos="http://cran.rstudio.org")
#install.packages("data.table")
library(feather)
library(h2o)
library(data.table)
library(stringr)
h2o.init(nthreads = 36, max_mem_size = "30G")
h2o.removeAll()

####################################################################
### Set-up the validation scheme                                 ###
####################################################################

train63d <- read_feather("../data/processed/end63_train.feather")
valid63d <- read_feather("../data/processed/end63_test.feather")


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
h2o.saveModel(object = retrained_dl_all, path = "../models/1stLevel/end77_dl_all", force=TRUE)
h2o.saveModel(object = retrained_dl_cat, path = "../models/1stLevel/end77_dl_cat", force=TRUE)

# save the predictions for the second level modeling
pred_all_train77d <- as.data.frame(h2o.predict(retrained_dl_all, newdata = retrain_set.hex))
pred_all_test77d <- as.data.frame(h2o.predict(retrained_dl_all, newdata = test_set.hex))
pred_all_train77d <- cbind(train77d_index_df, pred_all_train77d)
pred_all_test77d <- cbind(test77d_index_df, pred_all_test77d)
newnames = paste("preds_nn",i,sep="")
names(pred_all_train77d)[2] = newnames
names(pred_all_test77d)[2] = newnames

write_feather(pred_all_train77d, "../data/preds1stLevel/end77d_train_nn")
write_feather(pred_all_test77d, "../data/preds1stLevel/end77d_test_nn")

pred_cat_train77d <- as.data.frame(h2o.predict(retrained_dl_cat, newdata = retrain_set.hex))[,3]
pred_cat_test77d <- as.data.frame(h2o.predict(retrained_dl_cat, newdata = test_set.hex))[,3]
pred_cat_train77d <- cbind(train77d_index_df, pred_cat_train77d)
pred_cat_test77d <- cbind(test77d_index_df, pred_cat_test77d)
newnames = paste("nn",i,sep="")
names(pred_cat_train77d)[2] = newnames
names(pred_cat_test77d)[2] = newnames

write_feather(pred_cat_train77d, "../data/preds1stLevel/end77d_train_nncat")
write_feather(pred_cat_test77d, "../data/preds1stLevel/end77d_train_nncat")

# remove the data in h2o
h2o.rm(retrain_set.hex)
h2o.rm(test_set.hex)


####################################################################
### Retain the model on train92d                                 ###
####################################################################
#Load train92d and test92d dataset
train92d <- read_feather("../data/processed/end92_train.feather")
test92d <- read_feather("../data/processed/end92_test.feather")

train92d_index_df <- train92d[c("lineID")]
test92d_index_df <- test92d[c("lineID")]

#Load into the h2o environment
retrain_set.hex <-as.h2o(train92d[all_vars])
test_set.hex <-as.h2o(test92d[all_vars])
# factorize the categorical variables
for(c in cat_vars){
    retrain_set.hex[c] <- as.factor(retrain_set.hex[c])
}

for(c in cat_vars){
    test_set.hex[c] <- as.factor(test_set.hex[c])
}

rm(train92d, test92d)

#retain on the train92d dataset
retrained_dl_all <- do.call(h2o.deeplearning,
## update parameters in place
{
    p <- dl_all@parameters  # the same seed
    p$model_id = NULL          ## do not overwrite the original grid model
    p$training_frame = retrain_set.hex   ## use the full training dataset
    p$validation_frame = NULL  ## no validation frame
    p
})

sprintf("Model performance for the model train on all the features in train92d dataset")
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

sprintf("Model performance for the model train on categorical features in train92d dataset")
h2o.auc(h2o.performance(model = retrained_dl_cat, newdata = test_set.hex))

# save the models
h2o.saveModel(object = retrained_dl_all, path = "../models/1stLevel/end92_dl_all", force=TRUE)
h2o.saveModel(object = retrained_dl_cat, path = "../models/1stLevel/end92_dl_cat", force=TRUE)

# save the predictions for the second level modeling
pred_all_train92d <- as.data.frame(h2o.predict(retrained_dl_all, newdata = retrain_set.hex))
pred_all_test92d <- as.data.frame(h2o.predict(retrained_dl_all, newdata = test_set.hex))
pred_all_train92d <- cbind(train92d_index_df, pred_all_train92d)
pred_all_test92d <- cbind(test92d_index_df, pred_all_test92d)
newnames = paste("preds_nn",i,sep="")
names(pred_all_train92d)[2] = newnames
names(pred_all_test92d)[2] = newnames

write_feather(pred_all_train92d, "../data/preds1stLevel/end92d_train_nn")
write_feather(pred_all_test92d, "../data/preds1stLevel/end92d_test_nn")

pred_cat_train92d <- as.data.frame(h2o.predict(retrained_dl_cat, newdata = retrain_set.hex))[,3]
pred_cat_test92d <- as.data.frame(h2o.predict(retrained_dl_cat, newdata = test_set.hex))[,3]
pred_cat_train92d <- cbind(train92d_index_df, pred_cat_train92d)
pred_cat_test92d <- cbind(test92d_index_df, pred_cat_test92d)
newnames = paste("nn",i,sep="")
names(pred_cat_train92d)[2] = newnames
names(pred_cat_test92d)[2] = newnames

write_feather(pred_cat_train92d, "../data/preds1stLevel/end92d_train_nncat")
write_feather(pred_cat_test92d, "../data/preds1stLevel/end92d_train_nncat")


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