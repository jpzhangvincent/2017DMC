#!/usr/bin/env Rscript
#
# This script adds features that DO NOT depend on the labels:

LABEL_COLS <- c("click", "basket", "order", "revenue")

# The steps in this script are:
#
# 1. Compute nolabel features on items data.
#
# 2. Assemble lines data.
#     a. Remove labels from train data.
#     b. Stack train/test data.
#     c. Left join lines data with items data.
#
# 3. Compute nolabel features on lines data.
#
# 4. Partition lines data into CV sets, leader(board) set, and test set, as
#    shown below.
#
#     train / validation           leader   test
#     +----+----+----+----+----+   +----+   +----+
#     | 21 | 14 | 14 | 14 | 14 |   | 15 |   | 31 |
#     +----+----+----+----+----+   +----+   +----+
#         21   35   49   63   77       92
#
# 5. Write (train + leader) and test set to disk.
#

library(data.table)
library(feather)
library(stringr)

IN <- list(
  items = "../data/interim/1_clean_items.feather",
  train = "../data/interim/1_clean_train.feather",
  test  = "../data/interim/1_clean_test.feather")

OUT <- list(
  train = "../data/interim/2_nolabel_feat_train.feather",
  test  = "../data/interim/2_nolabel_feat_test.feather")

BREAKS <- c(21, 14, 14, 14, 14, 15)


# ==================== ITEMS ====================
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

message("Computed item features.")



# ==================== ASSEMBLE LINES ====================
train <- data.table(read_feather(IN$train))
test <- data.table(read_feather(IN$test))

# Split off labels.
labels <- train[, c("lineID", LABEL_COLS), with = FALSE]
train <- train[, !LABEL_COLS, with = FALSE]

# Stack line data & merge items data.
final_train_day <- max(train$day)
df <- rbind(train, test)
rm(train, test)

setkey(df, pid)
setkey(items, pid)
df <- merge(df, items, all.x = TRUE)
rm(items)

message("Assembled lines data.")



# ==================== NOLABEL FEATURES  ====================

# Day Cycles ----------------------------------------
df[, `:=`(
    day_mod_7  = day %% 7
    , day_mod_10 = day %% 10
    , day_mod_14 = day %% 14
    , day_mod_28 = day %% 28
    , day_mod_30 = day %% 30
  )]

# Impute competitorPrice ----------------------------------------
# Fill NAs with means from successively coarser groupings.
df[, `:=`(
    competitorPrice_is_na = as.integer(is.na(competitorPrice))
    , competitorPrice_imputed = competitorPrice
  )]
df[, competitorPrice_imputed :=
  ifelse(is.na(competitorPrice_imputed)
    , mean(competitorPrice, na.rm = TRUE)
    , competitorPrice_imputed)
  , by = pid]
df[, competitorPrice_imputed :=
  ifelse(is.na(competitorPrice_imputed)
    , mean(competitorPrice, na.rm = TRUE)
    , competitorPrice_imputed)
  , by = deduplicated_pid]
df[, competitorPrice_imputed :=
  ifelse(is.na(competitorPrice_imputed)
    , mean(competitorPrice, na.rm = TRUE)
    , competitorPrice_imputed)
  , by = .(group, content, unit, day_mod_7, salesIndex, adFlag)]
df[, competitorPrice_imputed :=
  ifelse(is.na(competitorPrice_imputed)
    , mean(competitorPrice, na.rm = TRUE)
    , competitorPrice_imputed)
  , by = .(group, content, unit)]
# Fill remaining NAs with price.
df[is.na(competitorPrice_imputed), competitorPrice_imputed := price]

# Delete competitorPrice column.
df[, competitorPrice := NULL]

# Price Per Unit ----------------------------------------
df[, `:=`(
    price_per_unit = price / total_units
    , competitorPrice_per_unit = competitorPrice_imputed / total_units
  )]

# Price Differences ----------------------------------------
df[, `:=`(
    price_diff = price - competitorPrice_imputed
    , price_discount = (price - rrp) / rrp
    , competitorPrice_discount = (competitorPrice_imputed - rrp) / rrp
  )]

df[, `:=`(
    is_lower_price = as.integer(price_diff < 0)
    , is_discount = as.integer(price_discount < 0)
    , price_discount_diff = price_discount - competitorPrice_discount
  )]

df[, isgreater_discount := as.integer(price_discount_diff > 0)]

df[, `:=`(
    price_discount_min = min(price_discount)
    , price_discount_p25 = quantile(price_discount, probs = 0.25)
    , price_discount_med = median(price_discount)
    , price_discount_p75 = quantile(price_discount, probs = 0.75)
    , price_discount_max = max(price_discount)
    , price_discount_mad = mad(price_discount)
    #, price_discount_sd = sd(price_discount)
  ), by = deduplicated_pid]

# Counts ----------------------------------------
# Count by level to get "popularity" feature for each product attribute
df[, content_d7cnt := log(.N), by = .(content, day_mod_7)]
df[, group_d7cnt := log(.N), by = .(group, day_mod_7)]
df[, manufacturer_d7cnt := log(.N), by =.(manufacturer, day_mod_7)]
df[, unit_d7cnt := log(.N), by = .(unit, day_mod_7)]
df[, pharmForm_d7cnt := log(.N), by = .(pharmForm, day_mod_7)]
df[, category_d7cnt := log(.N), by = .(category, day_mod_7)]
df[, campaignIndex_d7cnt := log(.N), by = .(campaignIndex, day_mod_7)]
df[, salesIndex_d7cnt := log(.N), by = .(salesIndex, day_mod_7)]

# Interaction summary statistics
df[, inter_gcucd7_cnt := log(.N),
  by = .(group, content, unit, campaignIndex, day_mod_7)]
df[, inter_gcucd10_cnt := log(.N),
  by = .(group, content, unit, adFlag, day_mod_10)]
df[, inter_gcucd30_cnt := log(.N),
  by = .(group, content, unit, availability, day_mod_30)]
df[, inter_gcuca_cnt := log(.N),
  by = .(group, content, unit, campaignIndex, availability, adFlag)]

# Price Trends ----------------------------------------
shift_mean = function(x, n, ...) Reduce("+", shift(x, 1:n, ...)) / n

setkey(df, lineID, pid)

# Is median ideal fill value?
df[, `:=`(
    prev_price = shift(price, 1, median(price))
    , prev5_price_avg = shift_mean(price, 5, price[[1]])
    , prev5_price_min = do.call(pmin, shift(price, 1:5, price[[1]]))
    , prev5_price_max = do.call(pmax, shift(price, 1:5, price[[1]]))

    , next_price = shift(price, 1, median(price), type = "lead")
    , next5_price_avg = shift_mean(price, 5, median(price), type = "lead")
    , next5_price_min = do.call(pmin, shift(price, 1:5, price[[1]]))
    , next5_price_max = do.call(pmax, shift(price, 1:5, price[[1]]))
  ), by = pid]

df[, `:=`(
    prev_price_pct_chg = (price - prev_price) / prev_price
    , prev5_price_diff = prev5_price_max - prev5_price_min
    , price_gt_prev5 = as.integer(price > prev5_price_avg)

    , next_price_pct_chg = (next_price - price) / price
    , next5_price_diff = next5_price_max - next5_price_min
    , price_lt_next5 = as.integer(price < next5_price_avg)
  )]

# Previous Product State ----------------------------------------
df[, `:=`(
    prev_availability = shift(availability, 1, "?")
    , prev_adFlag = shift(adFlag, 1, 0.5)
  ), by = pid]

# Interactions / Transitions
df[, `:=`(
    availability_trans = paste(prev_availability, availability, sep = "-")
    , adFlag_trans = paste(prev_adFlag, adFlag, sep ="-")
  )]

message("Computed nolabel features.")



# ==================== PARTITION LINES ====================
train <- df[day <= final_train_day]
test <- df[day > final_train_day]
rm(df)

setkey(train, lineID)
setkey(labels, lineID)
train <- merge(train, labels, all = TRUE)
rm(labels)

# NOTE: This is the only label feature generated in this file.
train[, `:=`(
    order_qty = as.integer(revenue / price)
    , fold = rep.int(seq_along(BREAKS), times = BREAKS)[day]
  )]

message("Partitioned lines data.")



# ==================== WRITE TO DISK ====================
write_feather(train, OUT$train)
message(sprintf("Wrote: %s", OUT$train))

write_feather(test, OUT$test)
message(sprintf("Wrote: %s", OUT$test))
