library(data.table)
library(stringr)
library(dummies)
library(feather)
library(zoo)

#--------------- Read Data ------------------------------------------
orig_train_df <- readRDS("data/interim/train_cleanformat.rds")
test_df <- fread('data/raw/class.csv', na.strings = c('', ' ', 'NA'))
item_df <- fread('data/raw/items.csv', na.strings = c('', ' ', 'NA'))
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
rm(item_df)


#---------------- create "purchase propensity" features for each pid -----------------
pid_purchase_info <- orig_train_df[,list(num_pid_click = log(sum(click, na.rm=T)+1), 
                                  mean_pid_click = log(mean(click, na.rm=T)+1),
                                  num_pid_basket = log(sum(basket, na.rm=T)+1), 
                                  mean_pid_basket = log(mean(basket, na.rm=T)+1),
                                  num_pid_order = log(sum(order, na.rm=T)+1),
                                  mean_pid_order = log(mean(order, na.rm=T)+1)
                                  ), by = pid]
# 100 new pids in the test set
# new_pids <- setdiff(test_df$pid, train_df$pid)
# integrate the "purchase propensity" features
combin_df <- merge(combin_df,pid_purchase_info, on = "pid", all.x = TRUE)
to_fix <- c("num_pid_click", "mean_pid_click", "num_pid_basket", "mean_pid_basket",
            "num_pid_order", "mean_pid_order")
# fix missing values resulted from the new pids in the test set
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, content, unit, availability, category, salesIndex, adFlag), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, availability, adFlag), .SDcols = to_fix]
rm(pid_purchase_info)

# "purchance one probability" feature
buy_one_info <- orig_train_df[, list(buy_one_prob = length(num_items_bought[num_items_bought==1])/.N), by = pid]
# "purchance more probability" feature
buy_more_info <- orig_train_df[, list(buy_more_prob = length(num_items_bought[num_items_bought>1])/.N), by = pid]
combin_df <- merge(combin_df, buy_one_info, on ="pid", all.x =TRUE)
combin_df <- merge(combin_df, buy_more_info, on = "pid", all.x=TRUE)
# similarily, need to fix missing values resulted from the new pids in the test set
to_fix <- c("buy_one_prob", "buy_more_prob")
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, content, unit, availability, category, salesIndex, adFlag), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, availability, adFlag), .SDcols = to_fix]
rm(buy_one_info)
rm(buy_more_info)


#---------------- create "time related" features ------------------------
combin_df[, `:=`(day_mod_7 = day%%7, day_mod_10 = day%%10, 
                 day_mod_14 = day%%14, day_mod_28 = day%%28, 
                 day_mod_30 = day%%30)]

orig_train_df[, `:=`(day_mod_7 = day%%7, day_mod_10 = day%%10, 
                 day_mod_14 = day%%14, day_mod_28 = day%%28, 
                 day_mod_30 = day%%30)]


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

# order propensity by day
pid_orderday7_info <- orig_train_df[, list(cnt_click_byday7 = log(sum(click, na.rm=T)+1),
                     cnt_basket_byday7 = log(sum(basket, na.rm=T)+1),
                     cnt_order_byday7 = log(sum(basket, na.rm=T)+1)), by = .(pid, day_mod_7)]
# integrate the "order propensity by day" features
combin_df <- merge(combin_df, pid_orderday7_info, by = c("pid", "day_mod_7"), all.x = TRUE)
to_fix <- c("cnt_click_byday7", "cnt_basket_byday7", "cnt_order_byday7")
# fix missing values resulted from the new pids in the test set
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, adFlag, salesIndex, campaignIndex, day_mod_7), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, adFlag, day_mod_7), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group), .SDcols = to_fix]
rm(pid_orderday7_info)

#--------------- create "price related" feature -----------------------
# price difference feature
combin_df[, price_diff := price-competitorPrice_imputed]
combin_df[, islower_price := as.integer(price_diff < 0)]
combin_df[, price_discount := (price - rrp)/rrp]
combin_df[, is_discount := as.integer(price_discount< 0)]
combin_df[, competitor_price_discount := (competitorPrice_imputed - rrp)/rrp]
combin_df[, price_discount_diff := price_discount - competitor_price_discount]
combin_df[, isgreater_discount := as.integer(price_discount_diff> 0)]

# "group, campaignIndex, salesIndex, adFlag" group
# represent the possibly similar product sector in a compaign
combin_df[, `:=`(max_price_disc = max(price_discount, na.rm=T),
              min_price_disc = min(price_discount, na.rm=T),
              var_price_disc = var(price_discount, na.rm=T),
              p25_price_disc = quantile(price_discount, probs = 0.25, na.rm=T),
              median_price_disc = median(price_discount, na.rm=T),
              p75_price_bygroup = quantile(price_discount, probs = 0.75, na.rm=T)
              ),
          by = .(group, campaignIndex, salesIndex, adFlag)]
# few missing values in  var_price_disc
to_fix <- c("max_price_disc", "min_price_disc", "var_price_disc", "p25_price_disc",
            "median_price_disc", "p75_price_bygroup")
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(var_price_disc), mean(x, na.rm=T), x)), 
                               by = .(group), .SDcols = to_fix]

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
combin_df[, total_units := content_part1*content_part2*content_part3]
combin_df[, rrp_per_unit := rrp/total_units]
combin_df[, price_per_unit := price/total_units]
combin_df[, competitorPrice_per_unit := competitorPrice_imputed/total_units]
rm(split_units)

# pricing trend feature (previous days/records, future days/records)
# Test Example: tt_df <- combin_df[pid %in% c('10898','10896'), .(pid, price, day)]
setkey(combin_df, pid, day)
combin_df[, last_price := shift(price, 1), by = pid]
combin_df$last_price[is.na(combin_df$last_price)] <- combin_df$price[is.na(combin_df$last_price)]
combin_df[, lprice_chg_pct := (price -last_price)/last_price]

#combin_df[, next_price := shift(price, 1, type = "lead"), by = pid][]

combin_df[, last3_price_avg := Reduce('+', shift(price, 1:3))/3, by = pid]
combin_df$last3_price_avg[is.na(combin_df$last3_price_avg)] <- combin_df$price[is.na(combin_df$last3_price_avg)]

#function(x,y){x[is.na(x)] <- 0; y[is.na(y)] <- 0; x + y}
combin_df[, last3_price_min := do.call(pmin, combin_df[, shift(price, 1:3), by = pid][,-1])]
combin_df[, last3_price_max := do.call(pmax, combin_df[, shift(price, 1:3), by = pid][,-1])]
combin_df$last3_price_min[is.na(combin_df$last3_price_min)] <- combin_df$price[is.na(combin_df$last3_price_min)]
combin_df$last3_price_max[is.na(combin_df$last3_price_max)] <- combin_df$price[is.na(combin_df$last3_price_max)]



#-------integrate "random effects" feature for high dimension categorical variables----
manufacturer_ref <- readRDS("data/interim/manufacturer_ref")
category_ref <- readRDS("data/interim/category_ref")
group_ref <- readRDS("data/interim/group_ref")
content_ref <- readRDS("data/interim/content_ref")
unit_ref <- readRDS("data/interim/unit_ref")
pharmForm_ref <- readRDS("data/interim/pharmForm_ref")
pid_ref <- readRDS("data/interim//pid_ref")

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
#or just use as.factor() -> sparse.matrix() in the R modeling pipeline
 

#--------------- Recover the data ---------------------------------------
train_df <- combin_df[day<63,]
write_feather(train_df, 'data/processed/training_set.feather')
valid_df <- combin_df[day>=63 & day<93,]
write_feather(valid_df, 'data/processed/validation_set.feather')
test_df <- combin_df[day>93,]
write_feather(test_df, 'data/processed/test_set.feather')