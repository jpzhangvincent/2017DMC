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
  items = "../data/interim/01_clean_items.feather",
  train = "../data/interim/01_clean_train.feather",
  test  = "../data/interim/01_clean_test.feather")

NA_STRINGS <- c("", " ", "NA")


# ==================== ITEMS ====================
if (!file.exists(OUT$items)) {
  items <- fread(IN$items, na.strings = NA_STRINGS)

  # Stringify --------------------
  str_cols <- c("manufacturer", "group", "content", "unit", "pharmForm",
    "salesIndex", "category", "campaignIndex")
  items[, (str_cols) := lapply(.SD, str_to_lower), .SDcols = str_cols]

  # Missing Values --------------------
  items[, `:=`(
      pharmForm_isNA = ifelse(is.na(pharmForm), 1, 0)
      , pharmForm = ifelse(is.na(pharmForm), "-999", pharmForm)

      , category_isNA = ifelse(is.na(category), 1, 0)
      , category = ifelse(is.na(category), "-999", category)

      , campaignIndex_isNA = ifelse(is.na(campaignIndex), 1, 0)
      , campaignIndex = ifelse(is.na(campaignIndex), "-999", campaignIndex)
    )]

  write_feather(items, OUT$items)
  message(sprintf("Wrote: %s", OUT$items))
}


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


if (!file.exists(OUT$train))
  clean_lines(IN$train, OUT$train, has_labels = TRUE)
if (!file.exists(OUT$test))
  clean_lines(IN$test, OUT$test)

