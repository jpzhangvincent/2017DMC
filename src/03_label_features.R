#!/usr/bin/env Rscript
#
# This script adds features that depend on the labels.
#
# Note (group, content, unit) groups similar products when filling missing
# values after grouping by pid.
#

library(data.table)
library(feather)

IN <- list(
  train = "../data/interim/02_nolabel_feat_train.feather",
  test  = "../data/interim/02_nolabel_feat_test.feather")

OUT <- list(
  train      = "../data/03_%s_train.feather",
  validation = "../data/03_%s_test.feather")

LABEL_COLS <- c("click", "basket", "order", "revenue")


# This function runs first when the script is sourced/executed.
main <- function() {
  train = data.table(read_feather(IN$train))

  n_folds = max(train$fold)
  for ( i in seq.int(2, n_folds) ) {
    if (i == n_folds) {
      name = "leader"
      message("Computing features for [1 ... ] [leader].")
    } else {
      name = sprintf("iter_%02i", i - 1)
      message(sprintf("Computing features for [1 ... ] [%i].", i))
    }

    # Separate folds into training and validation sets.
    tr = train[fold < i, ]
    vd = train[fold == i, !LABEL_COLS, with = FALSE]

    make_label_features(tr, vd, name)
    rm(tr, vd); gc()
  }

  message("Computing features for [train] [test].")
  
  test = data.table(read_feather(IN$test))
  make_label_features(train, test, "final")

  invisible (NULL)
}



make_label_features <- function(tr, vd, name) {

  log1sum = function(x) log(sum(x) + 1)

  # By pid ----------------------------------------
  setkey(tr, pid)
  setkey(vd, pid)

  by_pid <- tr[, .(
      # Order Propensities --------------------
      num_pid_click = log1sum(click)
      , prob_pid_click = mean(click)
      , num_pid_basket = log1sum(basket)
      , prob_pid_basket = mean(basket)
      , num_pid_order = log1sum(order)
      , prob_pid_order = mean(order)

      # Purchase Probabilities --------------------
      , buy_one_prob = length(order_qty[order_qty == 1]) / .N
      , buy_more_prob = length(order_qty[order_qty > 1]) / .N

      # Consecutive Order Probabilities --------------------
      , num_cons_orders =
        log1sum( (order == 1) & (order == shift(order, 1, 0)) )
      , prob_cons_orders =
        mean( (order == 1) & (order == shift(order, 1, 0)) )
    ), by = pid]

  tr <- merge(tr, by_pid, all.x = TRUE)
  vd <- merge(vd, by_pid, all.x = TRUE)
  rm(by_pid); gc()

  # By (pid, day_mod_7) ----------------------------------------
  setkey(tr, pid, day_mod_7)
  setkey(vd, pid, day_mod_7)

  by_pid_day7 = tr[, list(
      # Price Differences --------------------
      # Average price, price difference and discount difference for click,
      # basket, and order
      avg_price_click_info = mean(price[click])
      , avg_price_basket_info = mean(price[basket])
      , avg_price_order_info = mean(price[order])

      , avg_pricediff_click_info = mean(price_diff[click])
      , avg_pricediff_basket_info = mean(price_diff[basket])
      , avg_pricediff_order_info = mean(price_diff[order])

      , avg_pricediscdiff_click_info = mean(price_discount_diff[click])
      , avg_pricediscdiff_basket_info = mean(price_discount_diff[basket])
      , avg_pricediscdiff_order_info = mean(price_discount_diff[order])

      # Order Rates --------------------
      , cnt_click_byday7 = log1sum(click)
      , cnt_basket_byday7 = log1sum(basket)
      , cnt_order_byday7 = log1sum(order)
    ), by = .(pid, day_mod_7)]

  tr <- merge(tr, by_pid_day7, all.x = TRUE)
  vd <- merge(vd, by_pid_day7, all.x = TRUE)
  rm(by_pid_day7); gc()

  # By (...) ----------------------------------------
  group_cols = c("group", "content", "unit", "availability", "adFlag")

  by_group <- tr[, list(
      # Action Propensities --------------------
      click_propensity = mean(click) / (1 - mean(click))
      , basket_propensity = mean(basket) / (1 - mean(basket))
      , order_propensity = mean(order) / (1 - mean(order))
    ), by = group_cols]

  tr <- merge(tr, by_group, by = group_cols, all.x = TRUE)
  vd <- merge(vd, by_group, by = group_cols, all.x = TRUE)
  rm(by_group); gc()

  # Revenue ----------------------------------------
  by_group <- tr[, .(
      avg_revenue_by_group_7 = mean(revenue)
    ), by = c(group_cols, "day_mod_7")]

  tr <- merge(tr, by_group, by = c(group_cols, "day_mod_7"), all.x = T)
  vd <- merge(vd, by_group, by = c(group_cols, "day_mod_7"), all.x = T)
  rm(by_group); gc()

  by_group <- tr[, .(
      avg_revenue_by_group_10 = mean(revenue)
    ), by = c(group_cols, "day_mod_10")]

  tr <- merge(tr, by_group, by = c(group_cols, "day_mod_10"), all.x = T)
  vd <- merge(vd, by_group, by = c(group_cols, "day_mod_10"), all.x = T)
  rm(by_group); gc()

  by_group <- tr[, .(
      avg_revenue_by_group_30 = mean(revenue)
    ), by = c(group_cols, "day_mod_30")]

  tr <- merge(tr, by_group, by = c(group_cols, "day_mod_30"), all.x = T)
  vd <- merge(vd, by_group, by = c(group_cols, "day_mod_30"), all.x = T)
  rm(by_group); gc()


  # NOTE: This feature does not have high importance ranking and is expensive
  # to generate, so I've left it out.
  # Need to be careful about overfitting problem.
  #tr[, avg_revenue_per_pid_line := (
  #    (sum(revenue) - revenue) / (.N - 1) # leave-one-out
  #    + rnorm(.N, mean(revenue), 0.2)     # added noise
  #  ), by = pid]

  #vd[, avg_revenue_per_pid_line := NA]

  #vd[, AvgRevPerPidDay_adj :=
  #  ifelse(is.na(avg_revenue_per_pid_line)
  #    , mean(avg_revenue_per_pid_line, na.rm = T)
  #    , avg_revenue_per_pid_line)
  #tr[, mean(avg_revenue_per_pid_line),
  #  by = .(pid, campaignIndex, salesIndex, adFlag)]

  #combin_df[, AvgRevPerPidDay_adj := ifelse(is.na(AvgRevPerPidDay_adj),
  #    mean(AvgRevPerPidDay, na.rm=T), AvgRevPerPidDay_adj
  #  # Missing because new pid
  #  ), by = .(group, unit, content, campaignIndex, salesIndex, adFlag)]

  #combin_df[, AvgRevPerPidDay_adj := ifelse(is.na(AvgRevPerPidDay_adj),
  #    mean(AvgRevPerPidDay, na.rm=T), AvgRevPerPidDay_adj
  #  ), by = .(group, unit, content)]

  #combin_df[, AvgRevPerPidDay_adj := ifelse(is.na(AvgRevPerPidDay_adj),
  #    mean(AvgRevPerPidDay, na.rm=T), AvgRevPerPidDay_adj
  #  ), by = .(group)]

  #combin_df[, AvgRevPerPidDay := NULL]


  # Random Effects Encoding ----------------------------------------
  # TODO:

  # Write To Disk ----------------------------------------
  setkey(tr, lineID)
  out <- sprintf(OUT$train, name)
  write_feather(tr, out)
  message(sprintf("Wrote: %s", out))
  rm(tr); gc()

  setkey(vd, lineID)
  out <- sprintf(OUT$validation, name)
  write_feather(vd, out)
  message(sprintf("Wrote: %s", out))
  rm(vd); gc()

  invisible (NULL)
}


main()
