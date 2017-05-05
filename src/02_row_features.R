#!/usr/bin/env Rscript
#
# This script adds row-wise features to the data sets.
#

library(data.table)
library(feather)
library(stringr)

IN <- list(
  items = "../data/interim/01_clean_items.feather",
  train = "../data/interim/01_clean_train.feather",
  test  = "../data/interim/01_clean_test.feather")

OUT <- list(
  items = "../data/interim/02_row_items.feather",
  train = "../data/interim/02_row_train.feather",
  test  = "../data/interim/02_row_test.feather")


# ==================== ITEMS ====================
if (!file.exists(OUT$items)) {
  items <- data.table(read_feather(IN$items))

  # Deduplicate Items --------------------
  # NOTE: This must be computed before other features!
  items[, deduplicated_pid := pid[[1]], by = setdiff(colnames(items), "pid")]

  # Decode Content (00x00x00) --------------------
  split_units = str_split_fixed(items$content, "x", 3)
  dims = dim(split_units)
  split_units = str_replace_all(split_units, "[[:space:][:alpha:]]", "")
  split_units[split_units == ""] = "1"
  split_units = as.numeric(split_units)
  dim(split_units) = dims

  items[, `:=`(
      content_part1 = split_units[, 1]
      , content_part2 = split_units[, 2]
      , content_part3 = split_units[, 3]
    )]

  items[, total_units := content_part1 * content_part2 * content_part3]

  # Other Features --------------------
  items[, `:=`(
      group_begin_num = str_extract(group, "^[0-9]+")
      , rrp_per_unit = rrp / total_units
    )]

  write_feather(items, OUT$items)
  message(sprintf("Wrote: %s", OUT$items))
}

# ==================== TRAIN & TEST ====================
row_features_lines <- function(in_path, out_path, has_labels = FALSE) {
  df <- data.table(read_feather(in_path))

  df[, `:=`(
      day_mod_7  = day %% 7
      , day_mod_10 = day %% 10
      , day_mod_14 = day %% 14
      , day_mod_28 = day %% 28
      , day_mod_30 = day %% 30
    )]

  if (has_labels) {
    df[, `:=`(
        order_qty = as.integer(revenue / price)
      )]
  }

  write_feather(df, out_path)
  message(sprintf("Wrote: %s", out_path))
}


if (!file.exists(OUT$train))
  row_features_lines(IN$train, OUT$train, has_labels = TRUE)
if (!file.exists(OUT$test))
  row_features_lines(IN$test, OUT$test)
