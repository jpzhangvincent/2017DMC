#!/usr/bin/env Rscript
#
# This script adds features that depend on the labels.
#
# Note (group, content, unit) groups similar products when filling missing
# values after grouping by pid.
#

set.seed(260)

library(data.table)
library(feather)

source("3.4_likelihood_features.R")

IN <- list(
  train = "../data/interim/2_nolabel_feat_train.feather",
  test  = "../data/interim/2_nolabel_feat_test.feather")

OUT <- list(
  train      = "../data/interim/3_end%i_train.feather",
  validation = "../data/interim/3_end%i_test.feather")

LABEL_COLS <- c("click", "basket", "order", "revenue", "order_qty")


# This function runs first when the script is sourced/executed.
# Set `folds` to a vector of desired splits.
main <- function(folds) {
  train <- data.table(read_feather(IN$train))
  
  if (missing(folds))
    folds = seq.int(5, max(train$fold))

  for ( i in folds ) {
    end_tr <- max(train[fold == i - 1, day])
    end_vd <- max(train[fold == i, day])
    message(sprintf(
      "Computing features for [ 1 ... %i ] [ ... %i ].", end_tr, end_vd))
    
    # Separate folds into training and validation sets.
    df <- copy(train[fold <= i, ])
    
    make_label_features(df, i, end = end_tr)
    
    rm(df); gc()
  } 

  message("Computing features for [1 ... 92 ] [ ... ].")
  test <- data.table(read_feather(IN$test))
  test$fold <- 1000

  train <- rbind(train, test, fill = TRUE)
  rm(test); gc()

  make_label_features(train, 1000, end = 92)

  invisible (NULL)
}



make_label_features <- function(df, i, end) {

  log1sum = function(x) log(sum(x) + 1)

  # By pid ----------------------------------------
  by = "deduplicated_pid"
  setkeyv(df, by)

  oldcols = copy(colnames(df))
  df[fold < i, `:=`(
      # Order Propensities --------------------
      num_pid_click     = log1sum(click)
      , prob_pid_click  = mean(click)
      , num_pid_basket  = log1sum(basket)
      , prob_pid_basket = mean(basket)
      , num_pid_order   = log1sum(order)
      , prob_pid_order  = mean(order)

      # Purchase Probabilities --------------------
      , order_qty_eq_1_prob = length(order_qty[order_qty == 1]) / .N
      , order_qty_gt_1_prob = length(order_qty[order_qty > 1]) / .N

      # Consecutive Order Probabilities --------------------
      , num_cons_orders
        = log1sum( (order == 1) & (order == shift(order, 1, 0)) )
      , prob_cons_orders
        = mean( (order == 1) & (order == shift(order, 1, 0)) )
    ), by = by]

  to_fix = setdiff(colnames(df), oldcols)
  fill_label_features(df, to_fix, by)
  impute_label_features(df, to_fix)

  # By (pid, day_mod_7) ----------------------------------------
  by = c("deduplicated_pid", "day_mod_7")
  setkeyv(df, by)

  # Order Rates --------------------
  oldcols = copy(colnames(df))
  df[fold < i, `:=`(
      cnt_click_byday7  = log1sum(click)
      , cnt_basket_byday7 = log1sum(basket)
      , cnt_order_byday7  = log1sum(order)
    ), by = by]

  to_fix = setdiff(colnames(df), oldcols)
  fill_label_features(df, to_fix, by)
  impute_label_features(df, to_fix)

  # Price Differences --------------------
  # Average price, price difference and discount difference for click, basket,
  # and order

  # Click
  oldcols = copy(colnames(df))
  df[fold < i & click, `:=`(
      avg_price_click_info             = mean(price)
      , avg_price_diff_click_info      = mean(price_diff)
      , avg_price_disc_diff_click_info = mean(price_discount_diff)
    ), by = by]

  to_fix = setdiff(colnames(df), oldcols)
  fill_label_features(df, to_fix, by)
  impute_label_features(df, to_fix)

  # Basket
  oldcols = copy(colnames(df))
  df[fold < i & basket, `:=`(
       avg_price_basket_info            = mean(price)
      , avg_price_diff_basket_info      = mean(price_diff)
      , avg_price_disc_diff_basket_info = mean(price_discount_diff)
    ), by = by]

  to_fix = setdiff(colnames(df), oldcols)
  fill_label_features(df, to_fix, by)
  impute_label_features(df, to_fix)

  # Order
  oldcols = copy(colnames(df))
  df[fold < i & order, `:=`(
      avg_price_order_info             = mean(price)
      , avg_price_diff_order_info      = mean(price_diff)
      , avg_price_disc_diff_order_info = mean(price_discount_diff)
    ), by = by]

  to_fix = setdiff(colnames(df), oldcols)
  fill_label_features(df, to_fix, by)
  impute_label_features(df, to_fix)

  # By (...) ----------------------------------------
  by = c("group", "content", "unit", "availability", "adFlag")
  setkeyv(df, by)

  odds_ratio = function(x, trunc) {
    x = log1p( mean(x) / (1 - mean(x)) )
    x[is.infinite(x)] = trunc
    return (x)
  }

  oldcols = copy(colnames(df))
  df[fold < i, `:=`(
      # Action Propensities --------------------
      click_propensity    = odds_ratio(click, trunc = 400)
      , basket_propensity = odds_ratio(basket, trunc = 100)
      , order_propensity  = odds_ratio(order, trunc = 10)
    ), by = by]


  to_fix = setdiff(colnames(df), oldcols)
  fill_label_features(df, to_fix, by)
  impute_label_features(df, to_fix)

  # Revenue ----------------------------------------
  by = c("group", "content", "unit", "availability", "adFlag", "day_mod_7")
  setkeyv(df, by)

  oldcols = copy(colnames(df))
  df[fold < i, avg_revenue_by_group_7 := mean(revenue), by = by]

  to_fix = setdiff(colnames(df), oldcols)
  fill_label_features(df, to_fix, by)
  impute_label_features(df, to_fix)


  by = c("group", "content", "unit", "availability", "adFlag", "day_mod_10")
  setkeyv(df, by)

  oldcols = copy(colnames(df))
  df[fold < i, avg_revenue_by_group_10 := mean(revenue), by = by]

  to_fix = setdiff(colnames(df), oldcols)
  fill_label_features(df, to_fix, by)
  impute_label_features(df, to_fix)


  by = c("group", "content", "unit", "availability", "adFlag", "day_mod_30")
  setkeyv(df, by)

  oldcols = copy(colnames(df))
  df[fold < i, avg_revenue_by_group_30 := mean(revenue), by = by]

  to_fix = setdiff(colnames(df), oldcols)
  fill_label_features(df, to_fix, by)
  impute_label_features(df, to_fix)

  # Revenue Features ----------------------------------------
  oldcols = copy(colnames(df))
  df[fold < i,
      loo_mean_revenue_by_pid := loo_mean(revenue) + rnorm(.N, 0, 0.2)
    , by = "deduplicated_pid"]
  
  to_fix = setdiff(colnames(df), oldcols)
  fill_label_features(df, to_fix, by)
  impute_label_features(df, to_fix)

  # NOTE: This feature does not have high importance ranking and is expensive
  # to generate, so I've left it out.
  # Need to be careful about overfitting problem.
  #tr[, avg_revenue_per_pid_line := (
  #    (sum(revenue) - revenue) / (.N - 1) # leave-one-out
  #    + rnorm(.N, mean(revenue), 0.2)     # added noise
  #  ), by = pid]

  # Likelihood Encoding ----------------------------------------
  train <- df[fold < i, ]
  test <- df[fold == i, ]
  rm(df); gc()

  construct_likelihood(train, test)

  # Write To Disk ----------------------------------------
  setkey(train, lineID)
  setkey(test, lineID)

  out <- sprintf(OUT$train, end)
  write_feather(train, out)
  message(sprintf("Wrote: %s", out))

  out <- sprintf(OUT$validation, end)
  write_feather(test, out)
  message(sprintf("Wrote: %s", out))

  invisible (NULL)
}


# Fill label features using first in group.
fill_label_features <- function(df, to_fix, by) {
  df[, (to_fix) := lapply(.SD, function(x) na.omit(x)[1]),
    by = by, .SDcols = to_fix]

  invisible (NULL)
}

# Impute label features for novel pids.
impute_label_features <- function(df, to_fix) {
  df[, (to_fix)
      := lapply(.SD, function(x) ifelse(is.na(x), mean(x, na.rm = T), x)),
    by = "deduplicated_pid", .SDcols = to_fix]

  # 1st Attempt
  by <- c("group", "content", "unit", "adFlag", "salesIndex", "campaignIndex",
    "day_mod_7")

  df[, (to_fix)
      := lapply(.SD, function(x) ifelse(is.na(x), mean(x, na.rm = T), x)),
    by = by, .SDcols = to_fix]

  # 2nd Attempt
  by <- c("group", "content", "unit", "adFlag")

  df[, (to_fix)
      := lapply(.SD, function(x) ifelse(is.na(x), mean(x, na.rm = T), x)),
    by = by, .SDcols = to_fix]

  # 3rd Attempt
  df[, (to_fix)
      := lapply(.SD, function(x) ifelse(is.na(x), mean(x, na.rm = T), x)),
    by = .(group), .SDcols = to_fix]

  # 4th Attempt
  df[, (to_fix)
      := lapply(.SD, function(x) ifelse(is.na(x), mean(x, na.rm = T), x)),
    .SDcols = to_fix]

  invisible (NULL)
}


main()
