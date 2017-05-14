# NOTE: MAKE SURE TO SET A RANDOM SEED IN ANY SCRIPT THAT RUNS THIS SCRIPT.
#
# This script has functions to construct likelihood features.
#

library(data.table)
library(feather)


construct_likelihood = function(train, test) {
  # Combinations
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

  train[, `:=`(
      manu_group = NULL
      , content_unit_pharmForm = NULL
      , day_adFlag_availability_campaignIndex = NULL
    )]

  test[, `:=`(
      manu_group = NULL
      , content_unit_pharmForm = NULL
      , day_adFlag_availability_campaignIndex = NULL
    )]

  invisible (NULL)
}


combine = function(...) factor(paste(..., sep = "_"))


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
