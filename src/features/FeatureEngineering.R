library(data.table)
library(stringr)
library(dummies)

#--------------- Read Data ------------------------------------------
orig_train_df <- readRDS("notebooks/train_cleanformat.rds")
test_df <- fread('data/raw/class.csv', na.strings = c('', ' ', NA))
item_df <- fread('data/raw/items.csv', na.strings = c('', ' ', NA))
setkey(test_df, pid)
setkey(item_df, pid)
test_df <- merge(test_df, item_df, all.x = TRUE)
to_string_vars <- c("pid", "lineID", "manufacturer", "group", "content", "unit",
                    "pharmForm", "salesIndex", "category", "campaignIndex")
test_df <- test_df[, (to_string_vars) := lapply(.SD, as.character), .SDcols = to_string_vars]
test_df[, `:=`(group = tolower(group), pharmForm = tolower(pharmForm),unit = tolower(unit))]

# combine train and test data for some common feature engineering
train_df <- orig_train_df[, names(test_df), with = FALSE]
data_ls <- list(train_df, test_df)
combin_df <- rbindlist(data_ls)
rm(data_ls)


#---------------- create "purchase propensity" features for each pid -----------------
pid_purchase_info <- orig_train_df[,list(num_pid_click = log(sum(click, na.rm=T)+1), 
                                  mean_pid_click = log(mean(click, na.rm=T)+1),
                                  num_pid_basket = log(sum(basket, na.rm=T)+1), 
                                  mean_pid_basket = log(mean(basket, na.rm=T)+1),
                                  num_pid_order = log(sum(order, na.rm=T)+1),
                                  mean_pid_order = log(mean(order, na.rm=T)+1)
                                  ), by = pid]
# 100 new pids in the test set
new_pids <- setdiff(test_df$pid, train_df$pid)
# we choose means as the propensity features values for new pids in the test set
tmp_means <- data.table(t(colMeans(pid_purchase_info[,-1])))
new_pids_purchase_info <- cbind(data.table(pid = new_pids), tmp_means)
pid_purchase_info <- rbind(pid_purchase_info, new_pids_purchase_info)
setkey(pid_purchase_info, pid)
# integrate the "purchase propensity" features
combin_df <- combin_df[pid_purchase_info, on = "pid"]

# "purchance one probability" feature
buy_one_info <- orig_train_df[, list(buy_one_prob = length(num_items_bought[num_items_bought==1])/.N), by = pid]
# "purchance more probability" feature
buy_more_info <- orig_train_df[, list(buy_more_prob = length(num_items_bought[num_items_bought>1])/.N), by = pid]
combin_df <- combin_df[buy_one_info, on ="pid"]
combin_df <- combin_df[buy_more_info, on = "pid"]


#---------------- create "time related" features ------------------------
combin_df[, `:=`(day_mod_7 = day%%7, day_mod_10 = day%%10, 
                 day_mod_14 = day%%14, day_mod_28 = day%%28, day_mod_30 = day%%30)]


#---------------- indicate the "missing values" features ---------------------
# competitorPrice, pharmFormm, category and campaignIndex have missing values
combin_df[, lapply(.SD, function(x)sum(is.na(x)))] 

combin_df[, `:=`(pharmForm_isNA = ifelse(is.na(pharmForm) == TRUE, 1, 0),
              pharmForm = ifelse(is.na(pharmForm) == TRUE, "-999", pharmForm))]
combin_df[, `:=`(category_isNA = ifelse(is.na(category) == TRUE, 1, 0),
              category = ifelse(is.na(category) == TRUE, "-999", category))]
combin_df[, `:=`(campaignIndex_isNA = ifelse(is.na(campaignIndex) == TRUE, 1, 0),
              campaignIndex = ifelse(is.na(campaignIndex) == TRUE, "-999", campaignIndex))]
# impute the competitorPrice with
# first, the average price for each pid
combin_df[, competitorPrice_isNA := ifelse(is.na(competitorPrice) == TRUE, 1, 0)]
combin_df[, competitorPrice_imputed := ifelse(is.na(competitorPrice)==TRUE, 
                                              mean(competitorPrice, na.rm=T), competitorPrice),
          by = pid]
# still missing, replace with the average price group by the salesIndex, group, content, 
# adFlag, unit
combin_df[, competitorPrice_imputed := ifelse(is.na(competitorPrice_imputed)==TRUE, 
                                              mean(competitorPrice, na.rm=T), 
                                              competitorPrice_imputed),
          by = .(salesIndex, group, content, unit, adFlag)]
# still missing,  replace with the average price group by 
# day_mod_7, genericProduct, availability
combin_df[, competitorPrice_imputed := ifelse(is.na(competitorPrice_imputed)==TRUE, 
                                              mean(competitorPrice, na.rm=T), 
                                              competitorPrice_imputed),
          by = .(day_mod_7, genericProduct, availability)]
# no missing now!
combin_df[, competitorPrice:=NULL]


#---------------- create "aggregated statistics" features ---------------
# count by category indicate the "popularity" feature for a product
combin_df[, content_cnt := log(.N), by = content]
combin_df[, group_cnt := log(.N), by = group]
combin_df[, manufacturer_cnt := log(.N), by = manufacturer]
combin_df[, unit_cnt := log(.N), by = unit]
combin_df[, pharmForm_cnt := log(.N), by = pharmForm]
combin_df[, category_cnt := log(.N), by = category]
combin_df[, campaignIndex_cnt := log(.N), by = campaignIndex]
combin_df[, salesIndex_cnt := log(.N), by = salesIndex]



#--------------- create "price related" feature -----------------------
# price difference feature
combin_df[, price_diff := price-competitorPrice_imputed]
combin_df[, price_discount := (price - rrp)/rrp]
combin_df[, competitor_price_discount := (competitorPrice_imputed - rrp)/rrp]
combin_df[, price_discount_diff := price_discount - competitor_price_discount]

combin_df[, `:=`(max_price_bygroup = max(price, na.rm=T),
              min_price_bygroup = min(price, na.rm=T),
              var_price_bygroup = var(price, na.rm=T),
              p25_price_bygroup = quantile(price, probs = 0.25, na.rm=T),
              median_price_bygroup = median(price, na.rm=T),
              p75_price_bygroup = quantile(price, probs = 0.75, na.rm=T)
              ),
          by = .(group, category, unit, content, campaignIndex, salesIndex, adFlag)]

# price per unit feature
split_units = str_split_fixed(combin_df$content, "X", 3)
dims = dim(split_units)
split_units = str_replace_all(split_units, "[[:space:][:alpha:]]", "")
split_units[split_units == ""] = "1"
split_units = as.numeric(split_units)
dim(split_units) = dims
combin_df[, `:=`(content_part1 = split_units[, 1],
                 content_part2 = split_units[, 2],
                 content_part3 = split_units[, 3])]
combin_df[, total_unit := content_part1*content_part2*content_part3]
combin_df[, rrp_per_unit := rrp/total_unit]
combin_df[, price_per_unit := price/total_unit]
combin_df[, competitorPrice_per_unit := competitorPrice_imputed/total_unit]
rm(split_units)


# historical price difference and other m

#-------integrate "random effects" feature for high dimension categorical variables----
manufacturer_ref <- readRDS("data/processed/manufacturer_ref")
category_ref <- readRDS("data/processed/category_ref")
group_ref <- readRDS("data/processed/group_ref")
content_ref <- readRDS("data/processed/content_ref")
unit_ref <- readRDS("data/processed/unit_ref")
pharmForm_ref <- readRDS("data/processed/pharmForm_ref")
pid_ref <- readRDS("data/processed/pid_ref")

combin_df <- merge(combin_df, pid_ref, by = "pid", all.x = T)
combin_df <- merge(combin_df, manufacturer_ref, by = "manufacturer", all.x = T)
combin_df <- merge(combin_df, group_ref, by = "group", all.x = T)
combin_df <- merge(combin_df, category_ref, by = "category", all.x = T)
combin_df <- merge(combin_df, unit_ref, by = "unit", all.x = T)
combin_df <- merge(combin_df, pharmForm_ref, by = "pharmForm", all.x = T)
combin_df <- merge(combin_df, content_ref, by = "content", all.x = T)

fix_cols <- c("pid_ref", "manufacturer_ref", "category_ref","pharmForm_ref", "unit_ref", "content_ref")
combin_df[, (fix_cols):= lapply(.SD, function(x) ifelse(is.na(x)==TRUE, mean(x, na.rm = T), x)), 
          .SDcols = fix_cols]


#------- dummy categorical variable representation --------------
#combin_df[, lapply(.SD, function(x) length(unique(x))),.SDcols = to_string_vars]
#combin <- dummy.data.frame(combin_df, names = c("unit","salesIndex", "campainIndex"), sep = "_")
#or just factorize


#--------------- Recover the data ---------------------------------------
#train_df <- combin_df[day<63,]
#valid_df <- combin_df[day>=63 & day<93,]
#test_df <- combin_df[day>93,]