library(data.table)
library(tidyverse)
library(stringi)

# read data and treat empty strings as NA
train_df <- fread('../data/raw/train.csv', na.strings = c('', ' ', NA))
item_df <- fread('../data/raw/items.csv', na.strings = c('', ' ', NA)) 
setkey(train_df, pid)
setkey(item_df, pid)

orig_train_df <- merge(train_df, item_df, all.x = TRUE)

# convert categorical variables as string type
to_string_vars <- c("pid", "lineID", "manufacturer", "group", "content", "unit", 
                    "pharmForm", "genericProduct", "salesIndex", "category", "campaignIndex")
orig_train_df <- orig_train_df[, (to_string_vars) := lapply(.SD, as.character), .SDcols=to_string_vars]

# lowercase string to be consistent
orig_train_df[, `:=`(group = tolower(group), pharmForm = tolower(pharmForm),
                unit = tolower(unit))]

# get quantity 
orig_train_df[, num_items_bought := as.integer(revenue/price)]

saveRDS(orig_train_df, "train_cleanformat.rds")

