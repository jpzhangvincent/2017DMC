rm(list = ls())
library(xgboost)
library(dummies)
library(feather)
library(pROC)
library(data.table)
library(stringr)

####################################################################
### Set-up the validation scheme                                 ###
####################################################################

train <- read_feather("../data/processed/end63_train.feather")
valid <- read_feather("../data/processed/end63_test.feather")

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
if (REPLACE_HIGH_DIMENSION_VARS == TRUE) {
  cat_vars <- setdiff(cat_vars, HIGH_DIMENSION_VARS)
}


######create one-hot encoding


Create_One_Hot_Encoding = function(data,variable_name){
  new = dummy(data[[variable_name]], sep="_")
  colnames(new) = paste(variable_name,1:dim(new)[2],sep='_')
  return(new)
}
 
# factorize the categorical variables
train_OH = do.call(cbind,lapply(cat_vars, function(c) {Create_One_Hot_Encoding(train, c)}))

valid_OH = do.call(cbind,lapply(cat_vars, function(c) {Create_One_Hot_Encoding(valid, c)}))

train$order = as.factor(ifelse(train$order,1,0))
valid$order = as.factor(ifelse(valid$order,1,0))

label <- c("order", "order_qty")
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
max_depth =c(2, 4, 6, 8, 10),
subsample =c(0.2,0.4,0.6,0.8,1), 
colsample_bytree =c(0.2,0.4,0.6,0.8,1)
#gamma = c(0.2,0.4,0.6,0.8,1)
)

X_train = train[,names(train) %in% predictors]
y_train = as.numeric(levels(train$order))[train$order]

X_val = valid[,names(valid) %in% predictors]
y_val =  as.numeric(levels(valid$order))[valid$order]

dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train)
dval <- xgb.DMatrix(as.matrix(X_val), label = y_val)
watchlist <- list(eval = dval, train = dtrain)

AUC_Hyperparameters <- apply(xgb_grid_1, 1, function(parameterList){

    #Extract Parameters to test
  params = list(
    eta = parameterList[["eta"]],
    nrounds = parameterList[["nrounds"]],
    colsample_bytree = parameterList[["colsample_bytree"]],
    max_depth = parameterList[["max_depth"]],
    subsample = parameterList[["subsample"]],
    eval_metric = 'auc',
    objective = 'binary:logistic'
    
  )


    model_xgb <- xgb.train(params, dtrain, nthread = 40, nrounds = 1e4, watchlist, early_stopping_rounds = 10, maximize = TRUE)

    train_pred = predict(model_xgb, dval, ntreelimit = model_xgb$bestInd)
    auc_score = as.numeric(auc(y_val,train_pred))

    return(data.frame(auc_score, subsample=parameterList[["subsample"]], 
                      colsample_bytree=parameterList[["colsample_bytree"]],
                      eta=parameterList[["eta"]],
                      max_depth = parameterList[["max_depth"]], 'n_estimate'=model_xgb$best_ntreelimit))

})

AUC_list = do.call(rbind, AUC_Hyperparameters)

model_list = head(AUC_list[order(AUC_list$auc_score,decreasing = TRUE),],3)



####################################################################

### Retain the model on train77d                                 ###

####################################################################

#Load train77d and test77d dataset

train77 <- read_feather("../data/processed/end77_train.feather")

valid77 <- read_feather("../data/processed/end77_test.feather")

train77d_index_df <- train77[c("lineID")]

test77d_index_df <- valid77[c("lineID")]


# factorize the categorical variables
train77_OH = do.call(cbind,lapply(cat_vars, function(c) {Create_One_Hot_Encoding(train77, c)}))

valid77_OH = do.call(cbind,lapply(cat_vars, function(c) {Create_One_Hot_Encoding(valid77, c)}))

train77$order = as.factor(ifelse(train77$order,1,0))
valid77$order = as.factor(ifelse(valid77$order,1,0))



train77 = cbind(train77, train77_OH)
valid77 = cbind(valid77, valid77_OH)
rm(train77_OH,valid77_OH)



X_train = train77[,names(train77) %in% predictors]
y_train = as.numeric(levels(train77$order))[train77$order]

X_val = valid77[,names(valid77) %in% predictors]
y_val =  as.numeric(levels(valid77$order))[valid77$order]

dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train)
dval <- xgb.DMatrix(as.matrix(X_val), label = y_val)
watchlist <- list(eval = dval, train = dtrain)

i=1
retrain_models <- apply(model_list, 1, function(parameterList){
  
  #Extract Parameters to test
  params = list(
    eta = parameterList[["eta"]],
    nrounds = 1000,
    colsample_bytree = parameterList[["colsample_bytree"]],
    max_depth = parameterList[["max_depth"]],
    subsample = parameterList[["subsample"]],
    eval_metric = 'auc',
    objective = 'binary:logistic'
    
  )
  
  
  model_xgb <- xgb.train(params, dtrain, nthread = 40, nrounds = 1e4, watchlist, early_stopping_rounds = 5, maximize = TRUE)
  
  train_pred = predict(model_xgb, dtrain, ntreelimit = model_xgb$bestInd)
  val_pred = predict(model_xgb, dval, ntreelimit = model_xgb$bestInd)
  
  preds_train77d <- cbind(train77d_index_df, train_pred)
  preds_test77d <- cbind(test77d_index_df, val_pred)
  
  auc_score = as.numeric(auc(y_val,val_pred))
  
  newnames = paste("xgboost",i,sep="")
  i = i+1
  names(preds_train77d)[2] = newnames
  
  names(preds_test77d)[2] = newnames

  # save the retrain model to regenerate the predictions for 2nd level modeling 
  
  # and possibly useful for ensemble
  
  xgb.dump(model_xgb, paste("../models/1stLevel/xgboost",auc_score,sep='-'), with_stats = TRUE)
  
  write_feather(preds_train77d, paste0("../data/preds1stLevel/xgboost_train77d-",auc_score,'.feather'))
  
  write_feather(preds_test77d, paste0("../data/preds1stLevel/xgboost_test77d-",auc_score,'.feather'))
})


  

  
  
  
  
  


