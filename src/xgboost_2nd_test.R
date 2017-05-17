rm(list = ls())
library(xgboost)
library(dummies)
library(feather)
library(data.table)
library(stringr)
setwd('D:/Dropbox/UCDavis/2017spring/DMC_2017_task/github/2017DMC/src')
#setwd('/Users/RayLJazz/Dropbox/UCDavis/2017spring/DMC_2017_task/github/2017DMC/src')
#setwd("C:/Users/Olivia/Desktop")

####################################################################
### Set-up the validation scheme                                 ###
####################################################################

train <- read_feather("end63_train_layer2.feather")

valid <- read_feather("end63_test_layer2.feather")



# define predictors from original features

features <- fread("feature_list_layer2.csv")

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
DAY_MOD =  c("day_mod_10", "day_mod_14", "day_mod_28", "day_mod_30")

REPLACE_HIGH_DIMENSION_VARS <- TRUE

if (REPLACE_HIGH_DIMENSION_VARS == TRUE){
  
  cat_vars <- setdiff(cat_vars, HIGH_DIMENSION_VARS)
  
}

REPLACE_DAY_MOD <- TRUE

if (REPLACE_DAY_MOD == TRUE){
  
  cat_vars <- setdiff(cat_vars, DAY_MOD)
  cont_vars = c(cont_vars, DAY_MOD)
  
}



label <- c("revenue")

all_preds <- c(cat_vars, cont_vars)

all_vars <- c(all_preds, label)


######create one-hot encoding


Create_One_Hot_Encoding = function(data,variable_name){
  new = dummy(data[[variable_name]], sep="_")
  colnames(new) = paste(variable_name,1:dim(new)[2],sep='_')
  return(new)
}

# factorize the categorical variables
train_OH = do.call(cbind,lapply(cat_vars, function(c) {Create_One_Hot_Encoding(train, c)}))

valid_OH = do.call(cbind,lapply(cat_vars, function(c) {Create_One_Hot_Encoding(valid, c)}))




cat_OH_vars = colnames(train_OH)
predictors <- c(cat_OH_vars,cont_vars)

all_vars <- c(predictors, label)

train = cbind(train, train_OH)
valid = cbind(valid, valid_OH)
rm(train_OH,valid_OH)


####################################################################
### modeling part - Grid Search                                 ###
####################################################################



# train xgboost model

xgb_grid_1 = expand.grid(
  objective = 'binary:logistic',
  nrounds = 1000,
  eta = 0.02,#c(0.01, 0.001, 0.0001),
  max_depth =c(4, 6),
  subsample =c(0.6,0.8), 
  colsample_bytree =c(0.6,0.8),
  tweedie_variance_power = seq(1.7,2.0,0.1),
  alpha = c(0.001, 0.01)
  
  #gamma = c(0.2,0.4,0.6,0.8,1) 
)

X_train = train[,names(train) %in% predictors]
y_train = train$revenue

X_val = valid[,names(valid) %in% predictors]
y_val =  valid$revenue

dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train)
dval <- xgb.DMatrix(as.matrix(X_val), label = y_val)
watchlist <- list(train = dtrain ,eval = dval)

AUC_Hyperparameters <- apply(xgb_grid_1, 1, function(parameterList){
  
  #Extract Parameters to test
  params = list(
    eta = parameterList[["eta"]],
    nrounds = parameterList[["nrounds"]],
    colsample_bytree = parameterList[["colsample_bytree"]],
    max_depth = parameterList[["max_depth"]],
    subsample = parameterList[["subsample"]],
    eval_metric = 'rmse',#'tweedie-nloglik',
    objective = 'reg:tweedie',
    tweedie_variance_power = parameterList[['tweedie_variance_power']]
    
  )
  
  print(params)
  model_xgb <- xgb.train(params, dtrain, nthread = 40, nrounds = 1e4, watchlist, early_stopping_rounds = 10, maximize = FALSE)
  
  rmse = as.numeric(model_xgb$best_score)
  
  return(data.frame(rmse, subsample=parameterList[["subsample"]], 
                    colsample_bytree=parameterList[["colsample_bytree"]],
                    eta=parameterList[["eta"]],
                    max_depth = parameterList[["max_depth"]], 'n_estimate'=model_xgb$best_ntreelimit,
                    tweedie_variance_power = parameterList[['tweedie_variance_power']]))
  
})

AUC_list = do.call(rbind, AUC_Hyperparameters)

model_list = head(AUC_list[order(AUC_list$auc_score,decreasing = TRUE),],3)



# remove the data in h2o
rm(train)

rm(valid)




####################################################################

### Retain the model on train77d                                 ###

####################################################################

#Load train77d and test77d dataset

train77 <- read_feather("../data/processed/end77_train_2nd.feather")

test77 <- read_feather("../data/processed/end77_test_2nd.feather")

train77d_index_df <- train77[c("lineID")]

test77d_index_df <- valid77[c("lineID")]



# factorize the categorical variables
train77_OH = do.call(cbind,lapply(cat_vars, function(c) {Create_One_Hot_Encoding(train77, c)}))

valid77_OH = do.call(cbind,lapply(cat_vars, function(c) {Create_One_Hot_Encoding(valid77, c)}))



train77 = cbind(train77, train77_OH)
valid77 = cbind(valid77, valid77_OH)
rm(train77_OH,valid77_OH)



X_train = train77[,names(train77) %in% predictors]
y_train = train77$revenue

X_val = valid77[,names(valid77) %in% predictors]
y_val =  train77$revenue

dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train)
dval <- xgb.DMatrix(as.matrix(X_val), label = y_val)
watchlist <- list(train = dtrain,eval = dval)

i=1
retrain_models <- apply(model_list, 1, function(parameterList){
  
  #Extract Parameters to test
  params = list(
    eta = parameterList[["eta"]],
    nrounds = 1000,
    colsample_bytree = parameterList[["colsample_bytree"]],
    max_depth = parameterList[["max_depth"]],
    subsample = parameterList[["subsample"]],
    eval_metric = 'rmse',
    objective = 'reg:tweedie'
    
  )
  
  
  model_xgb <- xgb.train(params, dtrain, nthread = 40, nrounds = 1e4, watchlist, early_stopping_rounds = 10, maximize = FALSE)
  
  train_pred = predict(model_xgb, dtrain, ntreelimit = model_xgb$bestInd)
  val_pred = predict(model_xgb, dval, ntreelimit = model_xgb$bestInd)
  
  preds_train77d <- cbind(train77d_index_df, train_pred)
  preds_test77d <- cbind(test77d_index_df, val_pred)
  
  rmse = as.numeric(model_xgb$best_score)
  
  newnames = paste("preds_xgboost",i,sep="")
  
  colnames(preds_train77d) = c('lineID',newnames)
  
  colnames(preds_test77d) =c('lineID',newnames)
  
  # save the retrain model to regenerate the predictions for 2nd level modeling 
  
  # and possibly useful for ensemble
  
  write_feather(preds_test77d, paste0("preds_2nd/end77d_test_xgboost_",rmse,'.feather'))
  i = i+1
})



####################################################################

### Retain the model on train92d                                 ###

####################################################################

#Load train77d and test77d dataset

train92 <- read_feather("../data/processed/end92_train_2nd.feather")

test92 <- read_feather("../data/processed/end92_test_2nd.feather")

train92d_index_df <- train92[c("lineID")]

test92d_index_df <- valid92[c("lineID")]

train92$order = as.factor(ifelse(train92$order,1,0))


# factorize the categorical variables
train92_OH = do.call(cbind,lapply(cat_vars, function(c) {Create_One_Hot_Encoding(train92, c)}))

valid92_OH = do.call(cbind,lapply(cat_vars, function(c) {Create_One_Hot_Encoding(valid92, c)}))



train92 = cbind(train92, train92_OH)
valid92 = cbind(valid92, valid92_OH)
rm(train92_OH,valid92_OH)



X_train = train92[,names(train92) %in% predictors]
y_train = train92$revenue

X_val = valid92[,names(valid92) %in% predictors]

dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train)
dval <- xgb.DMatrix(as.matrix(X_val))
watchlist <- list(train = dtrain)

i=1
retrain_models <- apply(model_list, 1, function(parameterList){
  
  #Extract Parameters to test
  params = list(
    eta = parameterList[["eta"]],
    nrounds = 1000,
    colsample_bytree = parameterList[["colsample_bytree"]],
    max_depth = parameterList[["max_depth"]],
    subsample = parameterList[["subsample"]],
    eval_metric = 'rmse',
    objective = 'reg:tweedie'
    
  )
  
  
  model_xgb <- xgb.train(params, dtrain, nthread = 40,watchlist, early_stopping_rounds = 10,nrounds = 1e4, maximize = FALSE)
  
  train_pred = predict(model_xgb, dtrain, ntreelimit = model_xgb$bestInd)
  val_pred = predict(model_xgb, dval, ntreelimit = model_xgb$bestInd)
  
  preds_train92d <- cbind(train92d_index_df, train_pred)
  preds_test92d <- cbind(test92d_index_df, val_pred)
  
  rmse = as.numeric(model_xgb$best_score)
  
  newnames = paste("preds_xgboost",i,sep="")
  
  colnames(preds_train92d) = c('lineID',newnames)
  
  colnames(preds_test92d) =c('lineID',newnames)
  
  # save the retrain model to regenerate the predictions for 2nd level modeling 
  
  # and possibly useful for ensemble
  
  write_feather(preds_test92d, paste0("preds_2nd/end92_test_xgboost_",rmse,'.feather'))
  i = i+1
})