
library(feather)

#setwd('D:/Dropbox/UCDavis/2017spring/DMC_2017_task/github/2017DMC/src')


train_set <- read_feather("data/processed/end63_train.feather")
pesudo_test_set <- read_feather("data/processed/end63_test.feather")


# likelihood generator function for training dataset
Likelihood_Train_Generator <- function(X_train, y_train, var_name, noise_sd = 0.02) {
  groupby = as.character(X_train[[var_name]])
  likelihood_train =  (tapply(y_train, groupby,sum)[groupby]-y_train)/(tapply(y_train, groupby,length)-1)[groupby]
  likelihood_train[is.na(likelihood_train)] = ((sum(y_train)-y_train)/(length(y_train)-1))[is.na(likelihood_train)]
  likelihood_train = pmin(likelihood_train * rnorm(nrow(X_train), 1, noise_sd), 1)
  return(likelihood_train)
}




# likelihood generator function for testing dataset
 Likelihood_Test_Generator <- function(X_train, y_train, X_test, var_name){
   groupby_train = as.character(X_train[[var_name]])
   groupby_test = as.character(X_test[[var_name]])
   new_object <- setdiff(groupby_test, groupby_train)
   temp <- rep(sum(y_train) / nrow(X_train), length(new_object))
   names(temp) <- new_object
   likelihood_test <- c(tapply(y_train, groupby_train, sum) / tapply(y_train, groupby_train, length), temp)[groupby_test]
   return(likelihood_test)
 }



#' construct likelihood list
likelihood_list <- c('pid',"manufacturer",'group')

for(likelihood_term in likelihood_list){
  LL_train = as.numeric(Likelihood_Train_Generator(train_set, train_set$order, likelihood_term))
  train_set <- cbind(train_set, LL_train)
  names(train_set)[names(train_set) == 'LL_train'] = paste(likelihood_term,'likelihood',sep='_')
  LL_valid = as.numeric(Likelihood_Test_Generator(train_set, train_set$order, pesudo_test_set, likelihood_term))
  pesudo_test_set <- cbind(pesudo_test_set, LL_valid)
  names(pesudo_test_set)[names(pesudo_test_set) == 'LL_valid'] = paste(likelihood_term,'likelihood',sep='_')
}

