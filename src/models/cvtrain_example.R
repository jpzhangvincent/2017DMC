library(feather)
library(dplyr)
library(xgboost)
options(dplyr.width = Inf)
library(h2o)


#----------------------setting cross validation -----------------------------
train_set <- read_feather("data/processed/training_set.feather") 
# create a weight column based on the quantity of items bought
train_set <- train_set %>% mutate(num_items_bought = revenue/price) %>% 
  mutate(weight_quantity = ifelse(num_items_bought==0, 
                                  num_items_bought+1, num_items_bought)) %>% 
  mutate(lineID = as.integer(lineID)) %>% arrange(lineID)
# the weight column is useful for weighted training to specify in the h2o's models

get_fold_number = function(x){
  if(x %in% 1:15)
    return(1)
  else if(x %in% 16:31)
    return(2)
  else if(x %in% 32:46)
    return(3)
  else if(x %in% 47:61)
    return(4)
  else
    return(5)
}
train_set <- train_set %>% mutate(fold_indicator = sapply(day, get_fold_number))


#----------------------prepare training data frame--------------------------
h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "8G")  #max mem size is the maximum memory to allocate to H2O
train_set.hex <- as.h2o(train_set, destination_frame = "train_set.hex")
# factorize the categorical variables
train_set.hex$order <- as.factor(train_set.hex$order)
train_set.hex$manufacturer <- as.factor(train_set.hex$manufacturer)
train_set.hex$pharmForm <- as.factor(train_set.hex$pharmForm)
train_set.hex$group <- as.factor(train_set.hex$group)
train_set.hex$unit <- as.factor(train_set.hex$unit)
train_set.hex$category <- as.factor(train_set.hex$category)
train_set.hex$campaignIndex <- as.factor(train_set.hex$campaignIndex)
train_set.hex$salesIndex <- as.factor(train_set.hex$salesIndex)
train_set.hex$adFlag <- as.factor(train_set.hex$adFlag)
train_set.hex$last_adFlag <- as.factor(train_set.hex$last_adFlag)
train_set.hex$availability <- as.factor(train_set.hex$availability)
train_set.hex$last_avaibility <- as.factor(train_set.hex$last_avaibility)
train_set.hex$group_beginNum <- as.factor(train_set.hex$group_beginNum)
train_set.hex$genericProduct <- as.factor(train_set.hex$genericProduct)
train_set.hex$content <- as.factor(train_set.hex$content)
train_set.hex$avaibility_transition <- as.factor(train_set.hex$avaibility_transition)
train_set.hex$adFlag_transition <- as.factor(train_set.hex$adFlag_transition)

# response and predictors
y <- "order"
predictors <- setdiff(names(train_set), c("pid", "lineID", "day","order", "basket", "click", "revenue",
                                          "num_items_bought", "weight_quantity", "fold_indicator", 
                                          "content_part1", "content_part2", "content_part3"))
print(predictors)

# Train & Cross-validate a GBM
gbm1 <- h2o.gbm(x = predictors,
                y = y,
                training_frame = train_set.hex,
                distribution = "bernoulli",
                ntrees = 500,
                max_depth = 8,
                col_sample_rate_per_tree = 0.9,
                min_rows = 3,
                learn_rate = 0.05,
                fold_column = "fold_indicator",
                stopping_metric = "AUC",
                stopping_rounds = 5,
                weights_column = "weight_quantity",
                keep_cross_validation_predictions = TRUE,
                seed = 1)
print(gbm1)
h2o.varimp(gbm1)
plot(gbm1)
h2o.saveModel(gbm1, "models/allfeatures_auc_gbm1_0.657")

gbm2 <-  h2o.gbm(x = predictors,
                     y = y,
                     training_frame = train_set.hex,
                     distribution = "bernoulli",
                     ntrees = 500,
                     max_depth = 8,
                     col_sample_rate_per_tree = 0.9,
                     min_rows = 5,
                     learn_rate = 0.05,
                     fold_column = "fold_indicator",
                     stopping_metric = "misclassification",
                     stopping_rounds = 5,
                     weights_column = "weight_quantity",
                     keep_cross_validation_predictions = TRUE,
                     seed = 1)
print(gbm2)
summary(gbm2)
plot(gbm2)
h2o.saveModel(gbm2, "models/allfeatures_misclass_xgb_0.66")
head(as.data.frame(h2o.varimp(gbm2)))
h2o.varimp(gbm2)

#random comment
#---------------------- evaluate on the validation set-----------
valid_set <- read_feather("data/processed/validation_set.feather") #93>day>77
valid_set.hex <- as.h2o(valid_set, destination_frame = "valid_set.hex")
valid_set.hex$order <- as.factor(valid_set.hex$order)
valid_set.hex$manufacturer <- as.factor(valid_set.hex$manufacturer)
valid_set.hex$pharmForm <- as.factor(valid_set.hex$pharmForm)
valid_set.hex$group <- as.factor(valid_set.hex$group)
valid_set.hex$unit <- as.factor(valid_set.hex$unit)
valid_set.hex$category <- as.factor(valid_set.hex$category)
valid_set.hex$campaignIndex <- as.factor(valid_set.hex$campaignIndex)
valid_set.hex$salesIndex <- as.factor(valid_set.hex$salesIndex)
valid_set.hex$adFlag <- as.factor(valid_set.hex$adFlag)
valid_set.hex$last_adFlag <- as.factor(valid_set.hex$last_adFlag)
valid_set.hex$availability <- as.factor(valid_set.hex$availability)
valid_set.hex$last_avaibility <- as.factor(valid_set.hex$last_avaibility)
valid_set.hex$group_beginNum <- as.factor(valid_set.hex$group_beginNum)
valid_set.hex$genericProduct <- as.factor(valid_set.hex$genericProduct)
valid_set.hex$content <- as.factor(valid_set.hex$content)
valid_set.hex$avaibility_transition <- as.factor(valid_set.hex$avaibility_transition)
valid_set.hex$adFlag_transition <- as.factor(valid_set.hex$adFlag_transition)

valid_perf <- h2o.performance(model=gbm2, newdata = valid_set.hex)
print(valid_perf) 
#auc 0.75; best accuracy: 0.763 with threshold 0.68
