#!/usr/bin/env Rscript

set.seed(260)

library(data.table)
library(feather)

# TODO: 77/92

IN = c(
  train = "../data/interim/3_end%i_train.feather"
  , test = "../data/interim/3_end%i_test.feather")

OUT = c(
  train = "../data/merge/likelihood_end%i_train.rds"
  , test = "../data/merge/likelihood_end%i_test.rds")

main = function(d = 63) {
  paths = sprintf(IN, d)
  train = data.table(read_feather(paths[1]))
  message(sprintf("Read: %s", paths[1]))

  test = data.table(read_feather(paths[2]))
  message(sprintf("Read: %s", paths[2]))

  # Combinations
  combine = function(...) factor(paste(..., sep = "_"))

  train[, `:=`(
      manu_group = combine(manufacturer, group)
      , content_unit_pharmForm = combine(content, unit, pharmForm)
      , day_adFlag_availability_campaignIndex
        = combine(day, adFlag, availability, campaignIndex)
    )]

  test[, `:=`(
      manu_group = combine(manufacturer, group)
      , content_unit_pharmForm = combine(content, unit, pharmForm)
      , day_adFlag_availability_campaignIndex
        = combine(day, adFlag, availability, campaignIndex)
    )]

  # Construct likelihood list
  lhood_list = c("deduplicated_pid", 'pid', "manufacturer", 'group',
    'pharmForm','salesIndex', 'manu_group', 'content_unit_pharmForm',
    'day_adFlag_availability_campaignIndex')
  
  for(term in lhood_list){
    set_train_lhood(train, "order", term)
    set_test_lhood(train, test, "order", term)
  }

  keep = c("pid", paste0(lhood_list, "_likelihood"))

  train = train[, keep, with = FALSE]
  test = test[, keep, with = FALSE]
  
  paths = sprintf(OUT, d)
  saveRDS(train, paths[1])
  message(sprintf("Wrote: %s", paths[1]))

  saveRDS(test, paths[2])
  message(sprintf("Wrote: %s", paths[2]))

  invisible (NULL)
}


likelihood = function(x) (sum(x) - x) / (length(x) - 1)


set_train_lhood = function(df, col, by, noise_sd = 0.02) {
  col = as.symbol(col)
  name = paste0(by, "_likelihood")

  df[, (name) := likelihood(eval(col)), by = by]

  # Impute NAs.
  is_na = which(is.na(df[[name]]))
  set(df, is_na, name, df[, likelihood(eval(col))][is_na] )

  # Multiply by noise.
  set(df, NULL, name, pmin(df[[name]] * rnorm(nrow(df), noise_sd), 1) )

  invisible (NULL)
}


set_test_lhood = function(tr, vd, col, by) {
  col = as.symbol(col)
  name = paste0(by, "_likelihood")

  ll = tr[, mean(eval(col)), by = by]
  i = match(vd[[by]], ll[[by]])

  vd[, (name) := ll[i, 2]]

  # Impute NAs.
  is_na = which(is.na(vd[[name]]))
  set(vd, is_na, name, vd[, mean(eval(col))][is_na] )

  invisible (NULL)
}


lapply(c(63, 77, 92), main)
