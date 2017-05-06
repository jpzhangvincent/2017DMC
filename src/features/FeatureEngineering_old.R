library(data.table)
library(stringr)
library(feather)
#library(zoo)
#library(dummies)

# Notes:
# 1. "group, content, unit" pair usually represent as a similar pair when grouping 
#     to replace missing numeric values in the end (after grouping by pid)
# 2. Features can be broken down in main categories
#     1.Time
#     2.order propensity by pid and group (count, probability and odd ratio)
#     3.missing value treatments (isNA)
#     4.aggregated statistics related to product popularity 
#     5.order rate/revenue per pid by "day" (related response variables)
#     6.decoding variables with special string pattern(i.e group, unit)
#     7.dynamic pricing(price diff, price change, discount diff and pricing trends.)
#     8.product state last time per pid(availability and adFlag last time)
#     9."random effects encoding" for high dimension categorical variables
#`    `
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
rm(data_ls,item_df,train_df,test_df)

#---------------- create "time related" features ------------------------
combin_df[, `:=`(day_mod_7 = day%%7, day_mod_10 = day%%10, 
                 day_mod_14 = day%%14, day_mod_28 = day%%28, 
                 day_mod_30 = day%%30)]

orig_train_df[, `:=`(day_mod_7 = day%%7, day_mod_10 = day%%10, 
                     day_mod_14 = day%%14, day_mod_28 = day%%28, 
                     day_mod_30 = day%%30)]


#---------------- create "order propensity" features for each pid -----------------
pid_purchase_info <- orig_train_df[,list(num_pid_click = log(sum(click, na.rm=T)+1), 
                                  prob_pid_click = mean(click, na.rm=T),
                                  num_pid_basket = log(sum(basket, na.rm=T)+1), 
                                  prob_pid_basket = mean(basket, na.rm=T),
                                  num_pid_order = log(sum(order, na.rm=T)+1),
                                  prob_pid_order = mean(order, na.rm=T)
                                  ), by = pid]
# 100 new pids in the test set
# new_pids <- setdiff(test_df$pid, train_df$pid)
# integrate the "order propensity per pid" features
combin_df <- merge(combin_df,pid_purchase_info, by = "pid", all.x = TRUE)
to_fix <- c("num_pid_click", "prob_pid_click", "num_pid_basket", "prob_pid_basket",
            "num_pid_order", "prob_pid_order")
# fix missing values resulted from the new pids in the test set
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, content, unit, availability, category, salesIndex, adFlag), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, content, unit, category, availability, adFlag), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, availability, adFlag), .SDcols = to_fix]
rm(pid_purchase_info, to_fix)

# "order propensity per general product state" features
group_purchase_info <- orig_train_df[,list( 
                                      or_group_click = mean(click, na.rm=T)/(1-mean(click, na.rm=T)),
                                      or_group_basket = mean(basket, na.rm=T)/(1-mean(basket, na.rm=T)),
                                      or_group_order = mean(order, na.rm=T)/(1-mean(order, na.rm=T))
), by = .(group, content, unit, availability, adFlag)]
combin_df <- merge(combin_df, group_purchase_info, 
                   by = c("group", "content", "unit", "availability", "adFlag"), all.x = TRUE)
# fix missing values 
to_fix <- c("or_group_click", "or_group_basket", "or_group_order")
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, content, unit, adFlag), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, adFlag), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T | x<0.01, 0, x)),
          by=.(group), .SDcols = to_fix] # bounded by 0
rm(group_purchase_info, to_fix)



# "purchance one probability" feature
buy_one_info <- orig_train_df[, list(buy_one_prob = length(num_items_bought[num_items_bought==1])/.N), by = pid]
# "purchance more probability" feature
buy_more_info <- orig_train_df[, list(buy_more_prob = length(num_items_bought[num_items_bought>1])/.N), by = pid]
combin_df <- merge(combin_df, buy_one_info, by ="pid", all.x =TRUE)
combin_df <- merge(combin_df, buy_more_info, by = "pid", all.x=TRUE)
# similarily, need to fix missing values resulted from the new pids in the test set
to_fix <- c("buy_one_prob", "buy_more_prob")
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, content, unit, availability, category, salesIndex, adFlag), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, content, unit, category, availability, adFlag), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, availability, adFlag), .SDcols = to_fix]
rm(buy_one_info, buy_more_info)


# "probability of consectutive orders(this time and last time for same pid)" feature
setkey(orig_train_df, pid, day)
orig_train_df[, order_lasttime := shift(order, 1), by = pid]
orig_train_df$order_lasttime[is.na(orig_train_df$order_lasttime)] <- 0
orig_train_df[, is_consectutive_order := mapply(function(x,y){
                ifelse(x==1 & y==1, 1,0)}, order, order_lasttime), 
              by = pid]
cons_order_info <- orig_train_df[, .(num_cons_orders = log(sum(is_consectutive_order)+1),
                  prob_cons_orders = mean(is_consectutive_order)), by = pid]
# integrate the feature with the combin_df
combin_df <- merge(combin_df, cons_order_info, by ="pid", all.x =TRUE)
to_fix <- c("num_cons_orders", "prob_cons_orders")
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, content, unit, availability, category, salesIndex, adFlag), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, content, unit, category, availability, adFlag), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, availability, adFlag), .SDcols = to_fix]
rm(cons_order_info, to_fix)


#----------------deal with "missing values" features ---------------------
# competitorPrice, pharmFormm, category and campaignIndex have missing values
combin_df[, lapply(.SD, function(x)sum(is.na(x)))] 

combin_df[, `:=`(pharmForm_isNA = ifelse(is.na(pharmForm) == TRUE, 1, 0),
              pharmForm = ifelse(is.na(pharmForm) == TRUE, "-999", pharmForm))]
combin_df[, `:=`(category_isNA = ifelse(is.na(category) == TRUE, 1, 0),
              category = ifelse(is.na(category) == TRUE, "-999", category))]
combin_df[, `:=`(campaignIndex_isNA = ifelse(is.na(campaignIndex) == TRUE, 1, 0),
              campaignIndex = ifelse(is.na(campaignIndex) == TRUE, "-999", campaignIndex))]

# impute the missing "competitorPrice" features with
# first, the average price for each pid
combin_df[, competitorPrice_isNA := ifelse(is.na(competitorPrice) == TRUE, 1, 0)]
combin_df[, competitorPrice_imputed := ifelse(is.na(competitorPrice)==TRUE, 
                                              mean(competitorPrice, na.rm=T), competitorPrice),
          by = pid]
# still missing, replace with the average price group by the salesIndex, group, content, 
# adFlag, unit, day_mod_7
combin_df[, competitorPrice_imputed := ifelse(is.na(competitorPrice_imputed)==TRUE, 
                                              mean(competitorPrice, na.rm=T), 
                                              competitorPrice_imputed),
          by = .(group, day_mod_7, salesIndex, content, unit, adFlag)]
# still missing,  replace with the average price group by 
# group, content, unit
combin_df[, competitorPrice_imputed := ifelse(is.na(competitorPrice_imputed)==TRUE, 
                                              mean(competitorPrice, na.rm=T), 
                                              competitorPrice_imputed),
          by = .(group, content, unit)]
# if missing, set it to original price (more conservative)
combin_df[, competitorPrice_imputed := ifelse(is.na(competitorPrice_imputed)==TRUE, 
                                              price, 
                                              competitorPrice_imputed)]
combin_df[, competitorPrice:=NULL]

#---------------- create "aggregated statistics" features ---------------
# count by category indicate the "popularity" feature for a product attribute
combin_df[, content_cnt := log(.N), by = content]
combin_df[, group_cnt := log(.N), by = group]
combin_df[, manufacturer_cnt := log(.N), by = manufacturer]
combin_df[, unit_cnt := log(.N), by = unit]
combin_df[, pharmForm_cnt := log(.N), by = pharmForm]
combin_df[, category_cnt := log(.N), by = category]
combin_df[, campaignIndex_cnt := log(.N), by = campaignIndex]
combin_df[, salesIndex_cnt := log(.N), by = salesIndex]

# interaction summary statistics table 
combin_df[, inter_gcucd7_cnt := log(.N), by = .(group, content, unit, campaignIndex, day_mod_7)]
combin_df[, inter_gcucd10_cnt := log(.N), by = .(group, content, unit, adFlag, day_mod_10)]
combin_df[, inter_gcucd30_cnt := log(.N), by = .(group, content, unit, availability, day_mod_30)]
combin_df[, inter_gcuca_cnt := log(.N), by = .(group, content, unit, campaignIndex, availability, adFlag)]
#can be more...

# revenue by day 
AvgRevByAdDay7_info <- orig_train_df[, .(AvgRevByAdDay7 = mean(revenue)), by = .(group, content, unit, adFlag, availability, day_mod_7)]
AvgRevByAdDay10_info <- orig_train_df[, .(AvgRevByAdDay10 = mean(revenue)), by = .(group, content, unit, adFlag, availability, day_mod_10)]
AvgRevByAdDay30_info <- orig_train_df[, .(AvgRevByAdDay30 = mean(revenue)), by = .(group, content, unit, adFlag, availability, day_mod_30)]

combin_df <- merge(combin_df, AvgRevByAdDay7_info, by = c("group", "content", "unit", "adFlag", "availability", 
                                             "day_mod_7"), all.x=T)
combin_df <- merge(combin_df, AvgRevByAdDay10_info, by = c("group", "content", "unit", "adFlag", "availability", 
                                                          "day_mod_10"), all.x=T)
combin_df <- merge(combin_df, AvgRevByAdDay30_info, by = c("group", "content", "unit", "adFlag", "availability", 
                                                          "day_mod_30"), all.x=T)
to_fix <- c("AvgRevByAdDay7", "AvgRevByAdDay10", "AvgRevByAdDay30")
combin_df[, (to_fix):= lapply(.SD, function(x) ifelse(is.na(x)==TRUE, mean(x, na.rm = T), x)), 
          by = .(group, content, unit, adFlag, availability),
          .SDcols = to_fix]
combin_df[, (to_fix):= lapply(.SD, function(x) ifelse(is.na(x)==TRUE, mean(x, na.rm = T), x)), 
          by = .(group, content, unit),
          .SDcols = to_fix]
combin_df[, (to_fix):= lapply(.SD, function(x) ifelse(is.na(x)==TRUE, mean(x, na.rm = T), x)), 
          by = .(group),
          .SDcols = to_fix]
rm(AvgRevByAdDay7_info, AvgRevByAdDay10_info, AvgRevByAdDay30_info)

#----------------- create "order rate/revenue per pid by day" features
# "order propensity per pid by day" features
pid_orderday7_info <- orig_train_df[, list(cnt_click_byday7 = log(sum(click, na.rm=T)+1),
                     cnt_basket_byday7 = log(sum(basket, na.rm=T)+1),
                     cnt_order_byday7 = log(sum(basket, na.rm=T)+1)), by = .(pid, day_mod_7)]
# integrate the "order propensity by day" features
combin_df <- merge(combin_df, pid_orderday7_info, by = c("pid", "day_mod_7"), all.x = TRUE)
to_fix <- c("cnt_click_byday7", "cnt_basket_byday7", "cnt_order_byday7")
# fix missing values resulted from the new pids in the test set
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, content, unit, adFlag, salesIndex, campaignIndex, day_mod_7), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, content, unit, adFlag), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group), .SDcols = to_fix]
rm(pid_orderday7_info, to_fix)

# "revenue per pid by day" features
# need to be careful about overfitting problem -> use leave one out mean and add noise
# Ref: http://brooksandrew.github.io/simpleblog/articles/advanced-data-table/
orig_train_df[, AvgRevPerPidDay := (sum(revenue)-revenue)/(.N-1), by = pid]
orig_train_df[, AvgRevPerPidDay := sapply(AvgRevPerPidDay, function(x) ifelse(is.na(x)==T, 
                                                                              rnorm(1,mean(revenue,na.rm = T), 0.2), 
                                                                              AvgRevPerPidDay)), by = pid]
# join with the information from the training set
AvgRevPerPidDay_info <- orig_train_df[, .(lineID, AvgRevPerPidDay), with = T]
combin_df <- merge(combin_df, AvgRevPerPidDay_info, by = "lineID", all.x=T)
combin_df$AvgRevPerPidDay[combin_df$day>=93] <- NA
combin_df[, AvgRevPerPidDay_adj := ifelse(is.na(AvgRevPerPidDay)==T, mean(AvgRevPerPidDay, na.rm=T), AvgRevPerPidDay),
          by = .(pid, campaignIndex, salesIndex, adFlag)]   #by pid
combin_df[, AvgRevPerPidDay_adj := ifelse(is.na(AvgRevPerPidDay_adj)==T, mean(AvgRevPerPidDay, na.rm=T), AvgRevPerPidDay_adj),
          by = .(group, unit, content, campaignIndex, salesIndex, adFlag)] #missing because new pid
combin_df[, AvgRevPerPidDay_adj := ifelse(is.na(AvgRevPerPidDay_adj)==T, mean(AvgRevPerPidDay, na.rm=T), AvgRevPerPidDay_adj),
          by = .(group, unit, content)]
combin_df[, AvgRevPerPidDay_adj := ifelse(is.na(AvgRevPerPidDay_adj)==T, mean(AvgRevPerPidDay, na.rm=T), AvgRevPerPidDay_adj),
          by = .(group)]
combin_df[, AvgRevPerPidDay:=NULL]
rm(AvgRevPerPidDay_info)

#---------------- decode "string pattern" features from variables ------------------
# decode "group" variable: group always starts with a number 
combin_df[, group_beginNum:= sapply(group, function(x) str_extract(x,'[[:number:]]+'))]

# decode "unit" feature:  common --X--X--
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
rm(split_units)


#--------------- create "dynamic pricing" features -----------------------
# general "price difference" feature(each row)
combin_df[, price_diff := price-competitorPrice_imputed]
combin_df[, islower_price := as.integer(price_diff < 0)]
combin_df[, price_discount := (price - rrp)/rrp]
combin_df[, is_discount := as.integer(price_discount< 0)]
combin_df[, competitor_price_discount := (competitorPrice_imputed - rrp)/rrp]
combin_df[, price_discount_diff := price_discount - competitor_price_discount]
combin_df[, isgreater_discount := as.integer(price_discount_diff> 0)]

# average "price, price difference and discount difference" for click, basket and order
avg_price_click_info <- orig_train_df[click==1, .(avg_price_click = mean(price)), by = .(pid, day_mod_7)]
avg_price_basket_info <- orig_train_df[basket==1, .(avg_price_basket = mean(price)), by = .(pid, day_mod_7)]
avg_price_order_info <- orig_train_df[order==1, .(avg_price_order = mean(price)), by = .(pid, day_mod_7)]

orig_train_df[, price_diff := price-competitorPrice]
orig_train_df[, price_discount := (price-rrp)/rrp]
orig_train_df[, competitor_price_discount := (competitorPrice - rrp)/rrp]
orig_train_df[, price_discount_diff := price_discount-competitor_price_discount]

avg_pricediff_click_info <- orig_train_df[click==1, .(avg_pricediff_click = mean(price_diff)), by = .(pid, day_mod_7)]
avg_pricediff_basket_info <- orig_train_df[basket==1, .(avg_pricediff_basket = mean(price_diff)), by = .(pid, day_mod_7)]
avg_pricediff_order_info <- orig_train_df[order==1, .(avg_pricediff_order = mean(price_diff)), by = .(pid, day_mod_7)]
avg_pricediscdiff_click_info <- orig_train_df[click==1, .(avg_pricediscdiff_click = mean(price_discount_diff)), by = .(pid, day_mod_7)]
avg_pricediscdiff_basket_info <- orig_train_df[basket==1, .(avg_pricediscdiff_basket = mean(price_discount_diff)), by = .(pid, day_mod_7)]
avg_pricediscdiff_order_info <- orig_train_df[order==1, .(avg_pricediscdiff_order = mean(price_discount_diff)), by = .(pid, day_mod_7)]
# join back to the combin_df
combin_df <- merge(combin_df, avg_price_click_info, by=c("pid","day_mod_7"), all.x=T)
combin_df <- merge(combin_df, avg_price_basket_info, by=c("pid","day_mod_7"), all.x=T)
combin_df <- merge(combin_df, avg_price_order_info, by=c("pid","day_mod_7"), all.x=T)
combin_df <- merge(combin_df, avg_pricediff_click_info, by=c("pid","day_mod_7"), all.x=T)
combin_df <- merge(combin_df, avg_pricediff_basket_info, by=c("pid","day_mod_7"), all.x=T)
combin_df <- merge(combin_df, avg_pricediff_order_info, by=c("pid","day_mod_7"), all.x=T)
combin_df <- merge(combin_df, avg_pricediscdiff_click_info, by=c("pid","day_mod_7"), all.x=T)
combin_df <- merge(combin_df, avg_pricediscdiff_basket_info, by=c("pid","day_mod_7"), all.x=T)
combin_df <- merge(combin_df, avg_pricediscdiff_order_info, by=c("pid","day_mod_7"), all.x=T)
# impute missing values
#combin_df[, names(combin_df)[startsWith(names(combin_df), 'avg')] := NULL]
combin_df[, lapply(.SD, function(x)sum(is.na(x)))] 
to_fix <- names(combin_df)[startsWith(names(combin_df), 'avg')]
# fix missing values resulted from the new pids in the test set
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(pid, salesIndex, adFlag), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group, content, unit, adFlag), .SDcols = to_fix]
combin_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, mean(x, na.rm=T), x)),
          by=.(group), .SDcols = to_fix]
to_fix1 <- names(combin_df)[startsWith(names(combin_df), 'avg_price_')]
combin_df[, (to_fix1) := lapply(.SD, function(x) ifelse(is.na(x)==T, price[is.na(x)], x)), .SDcols = to_fix1]
to_fix2 <- setdiff(to_fix, to_fix1)
combin_df[, (to_fix2) := lapply(.SD, function(x) ifelse(is.na(x)==T, 0, x)), .SDcols = to_fix2]
rm(avg_price_click_info, avg_price_basket_info, avg_price_order_info, avg_pricediff_click_info,
   avg_pricediff_basket_info, avg_pricediff_order_info, avg_pricediscdiff_basket_info,
   avg_pricediscdiff_click_info,avg_pricediscdiff_order_info, to_fix, to_fix1, to_fix2)

# "group, content, unit, campaignIndex, salesIndex, adFlag" group
# represent the possibly similar product sector in a compaign
combin_df[, `:=`(max_price_disc = max(price_discount, na.rm=T),
              min_price_disc = min(price_discount, na.rm=T),
              p25_price_disc = quantile(price_discount, probs = 0.25, na.rm=T),
              median_price_disc = median(price_discount, na.rm=T),
              p75_price_bygroup = quantile(price_discount, probs = 0.75, na.rm=T)
              ),
          by = .(group, content, unit, campaignIndex, salesIndex, adFlag)]
combin_df[, diff_price_disc := max_price_disc - min_price_disc]

# "price per unit" feature
combin_df[, rrp_per_unit := rrp/total_units]
combin_df[, price_per_unit := price/total_units]
combin_df[, competitorPrice_per_unit := competitorPrice_imputed/total_units]

# pricing trend feature (previous days/records, future days/records)
# Test Example: tt_df <- combin_df[pid %in% c('10898','10896'), .(pid, price, day)]
setkey(combin_df, pid, day)
combin_df[, last_price := shift(price, 1), by = pid]
combin_df[, last_price := ifelse(is.na(last_price)==T, price, last_price)]
combin_df[, lprice_chg_pct := (price -last_price)/last_price]
combin_df[, next_price := shift(price, 1, type = "lead"), by = pid]
combin_df[, next_price := ifelse(is.na(next_price)==T, price, next_price)]
combin_df[, nprice_chg_pct := (next_price - price)/price]

combin_df[, last5_price_avg := Reduce('+', shift(price, 1:5))/5, by = pid]
combin_df[, last5_price_avg := ifelse(is.na(last5_price_avg)==T, price, last5_price_avg)]
combin_df[, last5_price_min := do.call(pmin, combin_df[, shift(price, 1:5), by = pid][,-1])]
combin_df[, last5_price_max := do.call(pmax, combin_df[, shift(price, 1:5), by = pid][,-1])]
combin_df$last5_price_min[is.na(combin_df$last5_price_min)] <- combin_df$price[is.na(combin_df$last5_price_min)]
combin_df$last5_price_max[is.na(combin_df$last5_price_max)] <- combin_df$price[is.na(combin_df$last5_price_max)]
combin_df[, last5_price_diff := last5_price_max - last5_price_min]
combin_df[, avglast5_price_isLower := as.integer(last5_price_avg<price)]

combin_df[, next5_price_avg := Reduce('+', shift(price, 1:5, type = "lead"))/5, by = pid]
combin_df[, next5_price_avg := ifelse(is.na(next5_price_avg)==T, price, next5_price_avg)]
combin_df[, next5_price_min := do.call(pmin, combin_df[, shift(price, 1:5, type = "lead"), by = pid][,-1])]
combin_df[, next5_price_max := do.call(pmax, combin_df[, shift(price, 1:5, type = "lead"), by = pid][,-1])]
combin_df[, next5_price_min := ifelse(is.na(next5_price_max)==T, price, next5_price_min)]
combin_df[, next5_price_max := ifelse(is.na(next5_price_max)==T, price, next5_price_max)]
combin_df[, next5_price_diff := next5_price_max - next5_price_min]
combin_df[, avgnext5_price_isLower := as.integer(price<next5_price_avg)]

#----------------add "product state last time" features---------------
setkey(combin_df, pid, day)
combin_df[, last_avaibility := shift(availability, 1), by = pid]
combin_df[, last_avaibility := ifelse(is.na(last_avaibility)==T, 5, last_avaibility)]
combin_df[, last_adFlag := shift(adFlag, 1), by = pid]
combin_df[, last_adFlag := ifelse(is.na(last_adFlag)==T, 5, last_adFlag)]

# interaction effect
combin_df[, avaibility_transition := paste(last_avaibility, availability, sep ='-')]
combin_df[, adFlag_transition := paste(last_adFlag, adFlag, sep ='-')]


#-------integrate "random effects" feature for encoding high dimension categorical variables----
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
rm(manufacturer_ref, category_ref, group_ref, content_ref, unit_ref, 
   pharmForm_ref,pid_ref)

#------- dummy categorical variable representation --------------
#combin_df[, lapply(.SD, function(x) length(unique(x))),.SDcols = to_string_vars]
#combin <- dummy.data.frame(combin_df, names = c("unit","salesIndex", "campainIndex"), sep = "_")
#or just use as.factor() -> sparse.matrix() in the R modeling pipeline
 

#-------- delete unused variables---------------------------------


#--------------- Recover the data ---------------------------------
train_df <- combin_df[day<= 77,]
tmp_train_label <- orig_train_df[day<=77, .(lineID, day, click, basket, order, revenue)]
train_df <- merge(tmp_train_label, train_df, by = c("lineID", "day"), all = T)
write_feather(train_df, 'data/processed/training_set.feather')

valid_df <- combin_df[day>77 & day<=92,]
tmp_valid_label <- orig_train_df[day>77 & day<=92, .(lineID, day, click, basket, order, revenue)]
valid_df <- merge(valid_df, tmp_valid_label, by = c("lineID", "day"), all = T)
write_feather(valid_df, 'data/processed/validation_set.feather')

test_df <- combin_df[day>=93,]
write_feather(test_df, 'data/processed/test_set.feather')
rm(train_df, valid_df, test_df)
rm(tmp_train_label, tmp_valid_label)

#------------- Create "nonlinear features" with deep learning ---------------
library(h2o)
train_set.hex <- as.h2o(train_df, destination_frame = "train_set.hex")
# factorize the categorical variables
train_set.hex$order <- as.factor(train_set.hex$order)
train_set.hex$manufacturer <- as.factor(train_set.hex$manufacturer)
train_set.hex$pharmForm <- as.factor(train_set.hex$pharmForm)
train_set.hex$group <- as.factor(train_set.hex$group)
train_set.hex$unit <- as.factor(train_set.hex$unit)
train_set.hex$category <- as.factor(train_set.hex$category)
train_set.hex$campaignIndex <- as.factor(train_set.hex$campaignIndex)
train_set.hex$salesIndex <- as.factor(train_set.hex$salesIndex)
train_set.hex$adFlag <- as.factor(train_set.hex$adFlag)
train_set.hex$last_adFlag <- as.factor(train_set.hex$last_adFlag)
train_set.hex$availability <- as.factor(train_set.hex$availability)
train_set.hex$last_avaibility <- as.factor(train_set.hex$last_avaibility)
train_set.hex$group_beginNum <- as.factor(train_set.hex$group_beginNum)
train_set.hex$genericProduct <- as.factor(train_set.hex$genericProduct)
train_set.hex$content <- as.factor(train_set.hex$content)
train_set.hex$avaibility_transition <- as.factor(train_set.hex$avaibility_transition)
train_set.hex$adFlag_transition <- as.factor(train_set.hex$adFlag_transition)
# response and predictors
y <- "order"
predictors <- setdiff(names(train_set), c("pid", "lineID", "day","order", "basket", "click", "revenue",
                                          "num_items_bought", "weight_quantity", "fold_indicator", 
                                          "content_part1", "content_part2", "content_part3"))
.dl = h2o.deeplearning(x = predictors, y = y, training_frame = train_set.hex,
                               hidden = c(100, 200), epochs = 5)
prostate.deepfeatures_layer1 = h2o.deepfeatures(prostate.dl, prostate.hex, layer = 1)
prostate.deepfeatures_layer2 = h2o.deepfeatures(prostate.dl, prostate.hex, layer = 2)
