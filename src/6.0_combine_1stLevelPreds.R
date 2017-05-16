#!/usr/bin/env Rscript

library(data.table)
library(feather)


IN = c(
  train = "../data/processed/end%i_train.feather"
  , test = "../data/processed/end%i_test.feather")


PRED_DIR = "../data/preds1stLevel"


main = function() {
  paths = list.files(PRED_DIR, "train77")

  train = data.table(read_feather())
  test = data.table(read_feather())
}

#main()
