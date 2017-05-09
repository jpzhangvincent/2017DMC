# H2O GLM
library(data.table)
library(feather)
library(h2o)
#library(h2oEnsemble)
h2o.shutdown()
h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "10G")  #max mem size is the maximum memory to allocate to H2O

train_set <- read_feather("~/Desktop/DMC_2017/training_set.feather")
valid_set <- read_feather("~/Desktop/DMC_2017/validation_set.feather")

train_set.hex <- as.h2o(train_set, destination_frame = "train_set.hex")
valid_set.hex <- as.h2o(valid_set, destination_frame = "valid_set.hex")

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


#valid_set.hex$order <- as.factor(valid_set.hex$order)
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


# split into train and pesudo validation set
# pesudo validation set is used to parameter tuning, using the last two weeks data from training set
train <- train_set.hex
valid <- valid_set.hex

# response and predictors
# Note: may need to update
response <- "order"
predictors <- setdiff(names(train_set), c("pid", "lineID", "day","order", "basket", "click", "revenue",
                                          "num_items_bought", "weight_qty", "fold_indicator", 
                                          "content_part1", "content_part2", "content_part3"))
print(predictors)


# GLM hyperparamters
alpha_opts = list(list(.0001), list(.00001),list(.000001))
lambda_opts = list(list(.0001), list(.00001),list(.000001))

glm_params = list(alpha = alpha_opts,lambda = lambda_opts)

# Random Grid Search
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

## save the top 5 models



###
## no need to retrain model. choose the best model and apply to all dataset
# Only choose the top 5 models and persist the retrained model
# Note: need to refit model including the pesudo validation set
for (i in 1:5) {
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
  retrained_glm_auc <- h2o.auc(h2o.performance(retrained_glm, newdata = valid_set.hex))
  preds <- h2o.predict(retrained_glm, valid_set.hex)
  # save the retrain model to regenerate the predictions for 2nd level modeling 
  # and possibly useful for ensemble
  h2o.saveModel(retrained_glm, paste("models/glm",glm@model_id,round(retrained_glm_auc,3),sep='-'), force=TRUE)
  h2o.exportFile(preds, paste0("data/interim/glm",glm@model_id,'1stLevelPred.csv',sep='-'), force=TRUE)
}

