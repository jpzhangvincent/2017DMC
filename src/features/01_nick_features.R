#!/usr/bin/env Rscript

library(stringr)

source("src/data/read_dmc.R")


train = read_dmc("data/train.csv")
items = read_dmc("data/items.csv")
test = read_dmc("data/class.csv")

train_out = train["lineID"]
items_out = items["pid"]
test_out = test["lineID"]


# total_units ----------------------------------------
message("total_units")
# `content` split on "X" and multiplied.

# Some rows have format ##X##X##.
split_units = str_split_fixed(items$content, "X", 3)

# Rows 5022, 5799 have leftover text and whitespace; strip this.
dims = dim(split_units)
split_units = str_replace_all(split_units, "[[:space:][:alpha:]]", "")
split_units[split_units == ""] = "1"
split_units = as.numeric(split_units)
dim(split_units) = dims

total_units = split_units[, 1] * split_units[, 2] * split_units[, 3]


# rrp_per_unit ----------------------------------------
message("rrp_per_unit")

rrp_per_unit = items$rrp / total_units


# price_per_unit ----------------------------------------
message("price_per_unit")
i = match(train$pid, items$pid)
j = match(test$pid, items$pid)

train_price_per_unit = train$price / total_units[i]
test_price_per_unit = test$price / total_units[j]


# competitorprice_per_unit ----------------------------------------
message("competitor_price_per_unit")

train_competitor_price_per_unit = train$competitorPrice / total_units[i]
test_competitor_price_per_unit = test$competitorPrice / total_units[j]


# day_of_week ----------------------------------------
message("day_of_week")

train_day_of_week = train$day %% 7
test_day_of_week = test$day %% 7


# OUTPUT ----------------------------------------

items_out$total_units = total_units
items_out$rrp_per_unit = rrp_per_unit
write_csv(test_out, "data/interim/items_rrp_per_unit.csv")

train_out$price_per_unit = train_price_per_unit
train_out$competitor_price_per_unit = train_competitor_price_per_unit
train_out$day_of_week = train_day_of_week
write_csv(train_out, "data/interim/train_price_per_unit.csv")

test_out$price_per_unit = test_price_per_unit
test_out$competitor_price_per_unit = test_competitor_price_per_unit
test_out$day_of_week = test_day_of_week
write_csv(test_out, "data/interim/test_price_per_unit.csv")

message("\nAll features saved!")
