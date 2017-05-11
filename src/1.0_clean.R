#!/usr/bin/env Rscript
#
# This script cleans the data sets without adding any new features.
#

library(data.table)
library(feather)
library(stringr)

IN <- list(
  items = "../data/raw/items.csv",
  train = "../data/raw/train.csv",
  test  = "../data/raw/class.csv")

OUT <- list(
  items = "../data/interim/1_clean_items.feather",
  train = "../data/interim/1_clean_train.feather",
  test  = "../data/interim/1_clean_test.feather")

NA_STRINGS <- c("", " ", "NA")


# ==================== ITEMS ====================
items <- fread(IN$items, na.strings = NA_STRINGS)

# Stringify --------------------
str_cols <- c("manufacturer", "group", "content", "unit", "pharmForm",
  "salesIndex", "category", "campaignIndex")
items[, (str_cols) := lapply(.SD, str_to_lower), .SDcols = str_cols]

# Missing Values --------------------
items[, `:=`(
    pharmForm_is_na = as.integer(is.na(pharmForm))
    , pharmForm = ifelse(is.na(pharmForm), "?", pharmForm)

    , category_is_na = as.integer(is.na(category))
    , category = ifelse(is.na(category), "?", category)

    , campaignIndex_is_na = as.integer(is.na(campaignIndex))
    , campaignIndex = ifelse(is.na(campaignIndex), "?", campaignIndex)
  )]

# Deduplicate Items --------------------
# NOTE: This must be computed before other features!
setkey(items, pid)

items[, deduplicated_pid := pid[[1]], by = setdiff(colnames(items), "pid")]

# Write --------------------
write_feather(items, OUT$items)
message(sprintf("Wrote: %s", OUT$items))


# ==================== TRAIN & TEST ====================
clean_lines <- function(in_path, out_path, has_labels = FALSE) {
  df <- fread(in_path, na.strings = NA_STRINGS)

  str_cols <- c("availability")
  df[, (str_cols) := lapply(.SD, as.character), .SDcols = str_cols]

  # Labels Only --------------------
  if (has_labels) {
    logi_cols <- c("click", "basket", "order")
    df[, (logi_cols) := lapply(.SD, as.logical), .SDcols = logi_cols]
  }

  write_feather(df, out_path)
  message(sprintf("Wrote: %s", out_path))
}


clean_lines(IN$train, OUT$train, has_labels = TRUE)
clean_lines(IN$test, OUT$test)

