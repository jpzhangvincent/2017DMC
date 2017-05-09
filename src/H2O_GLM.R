# H2O GLM
#
#This script is to train a glm model at the first level  
# to classify whether the product is ordered or not.
#
#Inputs: 
#     train_set(1<=day<=77): file from the path 'data/processed/train_set.feather'
#     pesudo_test_set(78<=day<=92): file from the path 'data/processed/validation_set.feather'
#Outputs: 
#     glm-*modelid*-*auc*: save the h2o models in the folder 'models/'
#     glm-*modelid*-1stLevelPred.csv: the prediction on the untouched test set


library(data.table)
library(feather)
library(h2o)
#library(h2oEnsemble)
h2o.shutdown()
h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "10G")  #max mem size is the maximum memory to allocate to H2O

train_set <- read_feather("~/Desktop/DMC_2017/training_set.feather")
pesudo_test_set <- read_feather("~/Desktop/DMC_2017/validation_set.feather")

train_set.hex <- as.h2o(train_set, destination_frame = "train_set.hex")
pesudo_test_set.hex <- as.h2o(pesudo_test_set, destination_frame = "pesudo_test_set.hex")

# factorize the categorical variables
#train_set.hex$order <- as.factor(train_set.hex$order)
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


#pesudo_test_set.hex$order <- as.factor(pesudo_test_set.hex$order)
pesudo_test_set.hex$manufacturer <- as.factor(pesudo_test_set.hex$manufacturer)
pesudo_test_set.hex$pharmForm <- as.factor(pesudo_test_set.hex$pharmForm)
pesudo_test_set.hex$group <- as.factor(pesudo_test_set.hex$group)
pesudo_test_set.hex$unit <- as.factor(pesudo_test_set.hex$unit)
pesudo_test_set.hex$category <- as.factor(pesudo_test_set.hex$category)
pesudo_test_set.hex$campaignIndex <- as.factor(pesudo_test_set.hex$campaignIndex)
pesudo_test_set.hex$salesIndex <- as.factor(pesudo_test_set.hex$salesIndex)
pesudo_test_set.hex$adFlag <- as.factor(pesudo_test_set.hex$adFlag)
pesudo_test_set.hex$last_adFlag <- as.factor(pesudo_test_set.hex$last_adFlag)
pesudo_test_set.hex$availability <- as.factor(pesudo_test_set.hex$availability)
pesudo_test_set.hex$last_avaibility <- as.factor(pesudo_test_set.hex$last_avaibility)
pesudo_test_set.hex$group_beginNum <- as.factor(pesudo_test_set.hex$group_beginNum)
pesudo_test_set.hex$genericProduct <- as.factor(pesudo_test_set.hex$genericProduct)
pesudo_test_set.hex$content <- as.factor(pesudo_test_set.hex$content)
pesudo_test_set.hex$avaibility_transition <- as.factor(pesudo_test_set.hex$avaibility_transition)
pesudo_test_set.hex$adFlag_transition <- as.factor(pesudo_test_set.hex$adFlag_transition)


# split into train and pesudo validation set
# pesudo validation set is used to parameter tuning, using the last two weeks data from training set
train <- train_set.hex[train_set.hex$day <= 63,]
valid <- train_set.hex[train_set.hex$day >63 & train_set.hex$day <=77,]

## try
train <- train_set.hex[train_set.hex$day <= 2,]
valid <- train_set.hex[train_set.hex$day ==3,]

# response and predictors
# Note: may need to update
response <- "order"
predictors <- setdiff(names(train_set), c("pid", "lineID", "day","order", "basket", "click", "revenue",
                                          "num_items_bought", "weight_qty", "fold_indicator", 
                                          "content_part1", "content_part2", "content_part3"))
print(predictors)


# GLM hyperparamters
alpha_opts = list(list(.0001), list(.000001))
lambda_opts = list(list(.5), list(.1), list(0.001))

glm_params = list(alpha = alpha_opts,lambda = lambda_opts)

# Random Grid Search
# no need 
search_criteria2 <- list(strategy = "RandomDiscrete", 
                         # train no more than 10 models
                         #max_models = 10,
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
                     #search_criteria = search_criteria,
                     x = predictors, 
                     y = response,
                     grid_id = "glm_grid1",
                     training_frame = train,
                     validation_frame = valid
                     #ntrees = 1000, learn_rate = 0.05, learn_rate_annealing = 0.99, weights_column = "weight_qty",
                     ## early stopping once the validation AUC doesn't improve 
                     #by at least 0.01% for 5 consecutive scoring events
                     #stopping_rounds = 5, 
                     #stopping_tolerance = 1e-4,
                     #stopping_metric = "AUC", 
                     #score_tree_interval = 10,
                     #seed = 1234
)

sorted_GLM_Grid <- h2o.getGrid(grid_id = "glm_grid1", 
                               sort_by = "auc", 
                               decreasing = TRUE)
print(sorted_GLM_Grid)
#glm_models <- lapply(glm_grid@model_ids, function(model_id) h2o.getModel(model_id))

# Only choose the top 5 models and persist the retrained model
# Note: need to refit model including the pesudo validation set
for (i in 1:2) {
  glm <- h2o.getModel(sorted_GLM_Grid@model_ids[[i]])
  retrained_glm <- do.call(h2o.glm,
                           ## update parameters in place
                           {
                             p <- glm@parameters  # the same seed
                             p$model_id = NULL          ## do not overwrite the original grid model
                             p$training_frame = train_set.hex   ## use the full training dataset
                             p$validation_frame = NULL  ## no validation frame
                             p
                           }
  )
  print(glm@model_id)
  ## Get the AUC on the hold-out test set
  retrained_glm_auc <- h2o.auc(h2o.performance(retrained_glm, newdata = pesudo_test_set.hex))
  preds <- h2o.predict(retrained_glm, pesudo_test_set.hex)
  # save the retrain model to regenerate the predictions for 2nd level modeling 
  # and possibly useful for ensemble
  h2o.saveModel(retrained_glm, paste("models/glm",glm@model_id,round(retrained_glm_auc,3),sep='-'), force=TRUE)
  h2o.exportFile(preds, paste0("data/interim/glm",glm@model_id,'1stLevelPred.csv',sep='-'), force=TRUE)
}

