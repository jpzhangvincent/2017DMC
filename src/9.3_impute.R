#
# This script has the imputation code used in scripts #3 and #4.
#

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
