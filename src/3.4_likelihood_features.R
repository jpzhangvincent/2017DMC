#!/usr/bin/env Rscript

library(feather)


train_set63 <- read_feather("../data/interim/3_end63_train.feather")
pesudo_test_set63 <- read_feather("../data/interim/3_end63_test.feather")

train_set77 <- read_feather("../data/interim/3_end77_train.feather")
pesudo_test_set77 <- read_feather("../data/interim/3_end77_test.feather")

# comnination of manu and group
train_set63$manu_group = factor(paste(train_set63$manufacturer, 
                                      train_set63$group,sep='_'))
pesudo_test_set63$manu_group = factor(paste(pesudo_test_set63$manufacturer, 
                                            pesudo_test_set63$group,sep='_'))

train_set63$manu_group_label = as.numeric(train_set63$manu_group)
pesudo_test_set63$manu_group_label = as.numeric(pesudo_test_set63$manu_group)

train_set77$manu_group = factor(paste(train_set77$manufacturer, 
                                      train_set77$group,sep='_'))
pesudo_test_set77$manu_group = factor(paste(pesudo_test_set77$manufacturer, 
                                            pesudo_test_set77$group,sep='_'))

train_set77$manu_group_label = as.numeric(train_set77$manu_group)
pesudo_test_set77$manu_group_label = as.numeric(pesudo_test_set77$manu_group)

# comnination of content, unit and pharmForm
train_set63$content_unit_pharmForm = factor(paste(train_set63$content, 
                                                  train_set63$unit, 
                                                  train_set63$pharmForm,sep='_'))
pesudo_test_set63$content_unit_pharmForm = factor(paste(pesudo_test_set63$content, 
                                                        pesudo_test_set63$unit, 
                                                        pesudo_test_set63$pharmForm ,sep='_'))

train_set63$content_unit_pharmForm_label =  as.numeric(train_set63$content_unit_pharmForm)
pesudo_test_set63$content_unit_pharmForm_label = as.numeric(pesudo_test_set63$content_unit_pharmForm)

train_set77$content_unit_pharmForm = factor(paste(train_set77$content, 
                                                  train_set77$unit, 
                                                  train_set77$pharmForm,sep='_'))
pesudo_test_set77$content_unit_pharmForm = factor(paste(pesudo_test_set77$content, 
                                                        pesudo_test_set77$unit, 
                                                        pesudo_test_set77$pharmForm ,sep='_'))

train_set77$content_unit_pharmForm_label =  as.numeric(train_set77$content_unit_pharmForm)
pesudo_test_set77$content_unit_pharmForm_label = as.numeric(pesudo_test_set77$content_unit_pharmForm)

# comnination of day, adflag, availability and campaignIndex
train_set63$day_adflag_availability_campaignIndex = factor(paste(train_set63$day, 
                                                                 train_set63$adFlag, 
                                                                 train_set63$availability, 
                                                                 train_set63$campaignIndex,sep='_'))
pesudo_test_set63$day_adflag_availability_campaignIndex = factor(paste(pesudo_test_set63$day, 
                                                                       pesudo_test_set63$adFlag, 
                                                                       pesudo_test_set63$availability, 
                                                                       pesudo_test_set63$campaignIndex,sep='_'))

train_set63$day_adflag_availability_campaignIndex_label =  as.numeric(train_set63$day_adflag_availability_campaignIndex)
pesudo_test_set63$day_adflag_availability_campaignIndex_label = as.numeric(pesudo_test_set63$day_adflag_availability_campaignIndex)

train_set77$day_adflag_availability_campaignIndex = factor(paste(train_set77$day, train_set77$adFlag, 
                                                                 train_set77$availability, 
                                                                 train_set77$campaignIndex,sep='_'))
pesudo_test_set77$day_adflag_availability_campaignIndex = factor(paste(pesudo_test_set77$day, 
                                                                       pesudo_test_set77$adFlag, 
                                                                       pesudo_test_set77$availability, 
                                                                       pesudo_test_set77$campaignIndex,sep='_'))

train_set77$day_adflag_availability_campaignIndex_label =  as.numeric(train_set77$day_adflag_availability_campaignIndex)
pesudo_test_set77$day_adflag_availability_campaignIndex_label = as.numeric(pesudo_test_set77$day_adflag_availability_campaignIndex)


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
likelihood_list <- c("deduplicated_pid",'pid',"manufacturer",'group','pharmForm','salesIndex',
                     'manu_group','content_unit_pharmForm','day_adflag_availability_campaignIndex')

for(likelihood_term in likelihood_list){
  LL_train = as.numeric(Likelihood_Train_Generator(train_set63, train_set63$order, likelihood_term))
  train_set63 <- cbind(train_set63, LL_train)
  names(train_set63)[names(train_set63) == 'LL_train'] = paste(likelihood_term,'likelihood',sep='_')
  LL_valid = as.numeric(Likelihood_Test_Generator(train_set63, train_set63$order, pesudo_test_set63, likelihood_term))
  pesudo_test_set63 <- cbind(pesudo_test_set63, LL_valid)
  names(pesudo_test_set63)[names(pesudo_test_set63) == 'LL_valid'] = paste(likelihood_term,'likelihood',sep='_')
}


for(likelihood_term in likelihood_list){
  LL_train = as.numeric(Likelihood_Train_Generator(train_set77, train_set77$order, likelihood_term))
  train_set77 <- cbind(train_set77, LL_train)
  names(train_set77)[names(train_set77) == 'LL_train'] = paste(likelihood_term,'likelihood',sep='_')
  LL_valid = as.numeric(Likelihood_Test_Generator(train_set77, train_set77$order, pesudo_test_set77, likelihood_term))
  pesudo_test_set77 <- cbind(pesudo_test_set77, LL_valid)
  names(pesudo_test_set77)[names(pesudo_test_set77) == 'LL_valid'] = paste(likelihood_term,'likelihood',sep='_')
}


likelihood_end63_train = train_set63[,names(train_set63) %in% c("pid","deduplicated_pid_likelihood","pid_likelihood", 
                                                                "manufacturer_likelihood","group_likelihood",
                                                                "pharmForm_likelihood","salesIndex_likelihood",
                                                                "manu_group_likelihood","content_unit_pharmForm_likelihood"               
                                                                ,"day_adflag_availability_campaignIndex_likelihood",
                                                                "manu_group_label","content_unit_pharmForm_label", 
                                                                "day_adflag_availability_campaignIndex_label")]
likelihood_end63_test = pesudo_test_set63[,names(pesudo_test_set63) %in% c("pid","deduplicated_pid_likelihood","pid_likelihood", 
                                                                           "manufacturer_likelihood","group_likelihood",
                                                                           "pharmForm_likelihood","salesIndex_likelihood",
                                                                           "manu_group_likelihood","content_unit_pharmForm_likelihood"               
                                                                           ,"day_adflag_availability_campaignIndex_likelihood",
                                                                           "manu_group_label","content_unit_pharmForm_label", 
                                                                           "day_adflag_availability_campaignIndex_label")]

likelihood_end77_train = train_set77[,names(train_set77) %in% c("pid","deduplicated_pid_likelihood","pid_likelihood", 
                                                                "manufacturer_likelihood","group_likelihood",
                                                                "pharmForm_likelihood","salesIndex_likelihood",
                                                                "manu_group_likelihood","content_unit_pharmForm_likelihood"               
                                                                ,"day_adflag_availability_campaignIndex_likelihood",
                                                                "manu_group_label","content_unit_pharmForm_label", 
                                                                "day_adflag_availability_campaignIndex_label")]
likelihood_end77_test = pesudo_test_set77[,names(pesudo_test_set77) %in% c("pid","deduplicated_pid_likelihood","pid_likelihood", 
                                                                           "manufacturer_likelihood","group_likelihood",
                                                                           "pharmForm_likelihood","salesIndex_likelihood",
                                                                           "manu_group_likelihood","content_unit_pharmForm_likelihood"               
                                                                           ,"day_adflag_availability_campaignIndex_likelihood",
                                                                           "manu_group_label","content_unit_pharmForm_label", 
                                                                           "day_adflag_availability_campaignIndex_label")]

write_feather(likelihood_end63_train, '../data/merge/likelihood_end63_train.feather')
write_feather(likelihood_end63_test, '../data/merge/likelihood_end63_test.feather')

write_feather(likelihood_end77_train, '../data/merge/likelihood_end77_train.feather')
write_feather(likelihood_end77_test, '../data/merge/likelihood_end77_test.feather')
