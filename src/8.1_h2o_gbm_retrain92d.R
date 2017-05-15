#!/usr/bin/env Rscript

library(data.table)
library(feather)

#load the 92day training data


#load the 1st level models in the "models/1stLevel/" folder


#retrain the 1st level models with 92day training data


#generate the predictions for 2nd modeling


#save the retrained 1st models in the "models/retrained_1stLmodels"


#load the 2nd level models in the "models/2ndLevel/" folder


#combine with the other features


#retrain the 2nd level models with 92day training data


#save the retrained models in the "models/retrained_2ndmodels/"


#load the test set "test92d.feather"


#generate predictions on the test set


#save the predictions in the "data/final_test_preds/h2o_gbm_pred.csv"