library(data.table)
library(glmmTMB) 

# Reasons for using generalized mixed effect model
# - high dimensionality of categorical variable, so it's better to 
#   infer and quantify the impacts of different levels 
# - noval levels in test set, so treating the variable as random effect sounds intuitive
# but in the end, we would rely on the feature election to verify this approach 

orig_train_df <- readRDS("data/interim/train_cleanformat.rds")
train_df <- orig_train_df[day<=77]
train_df[, `:=`(day_mod_7 = day%%7, day_mod_10 = day%%10, 
                     day_mod_14 = day%%14, day_mod_28 = day%%28, 
                     day_mod_30 = day%%30)]
orig_train_df[, `:=`(day_mod_7 = day%%7, day_mod_10 = day%%10, 
                day_mod_14 = day%%14, day_mod_28 = day%%28, 
                day_mod_30 = day%%30)]
#valid_df <- orig_train_df[day>62]

string_vars <- c("pid", "lineID", "manufacturer", "group", "content", "unit", "pharmForm", 
                 "genericProduct", "salesIndex", "category", "campaignIndex")
train_df[, lapply(.SD, function(x) length(unique(x))),.SDcols = string_vars]

# encode pid
encode_pid_mxef <- glmmTMB(num_items_bought ~ (1|pid),
                           data= train_df, ziformula = ~1,
                           family= poisson)
pid_ref <- ranef(encode_pid_mxef)$cond$pid
setDT(pid_ref, keep.rownames = TRUE)
names(pid_ref) <- c("pid", "pid_ref")
saveRDS(pid_ref, "data/interim/pid_ref")
rm(encode_pid_mxef)
# 
# # encode pid and day_mod_7 interaction effect
encode_pid_day7_mxef <- glmmTMB(num_items_bought ~ (1|day_mod_7/pid),
                           data= train_df, ziformula = ~1,
                           family= poisson)
pid_day7_ref <- ranef(encode_pid_day7_mxef)$cond$pid
setDT(pid_day7_ref, keep.rownames = TRUE)
names(pid_day7_ref) <- c("pid_day_mod_7", "pid_day7_ref")
saveRDS(pid_day7_ref, "data/interim/pid_day7_inter_ref")
day7_ref <- ranef(encode_pid_day7_mxef)$cond$day_mod_7
setDT(day7_ref, keep.rownames = TRUE)
names(day7_ref) <- c("day_mod_7", "day7_ref")
saveRDS(day7_ref, "data/interim/pid_day7_ref")
rm(encode_pid_day7_mxef)
# 
# # encode pid and day_mod_10 interaction effect
encode_pid_day10_mxef <- glmmTMB(num_items_bought ~ (1|day_mod_10/pid),
                                data= train_df, ziformula = ~1,
                                family= poisson)
pid_day10_ref <- ranef(encode_pid_day10_mxef)$cond$pid
setDT(pid_day10_ref, keep.rownames = TRUE)
names(pid_day10_ref) <- c("pid_day_mod_10", "pid_day10_ref")
saveRDS(pid_day10_ref, "data/interim/pid_day10_inter_ref")
day10_ref <- ranef(encode_pid_day10_mxef)$cond$day_mod_10
setDT(day10_ref, keep.rownames = TRUE)
names(day10_ref) <- c("day_mod_10", "day10_ref")
saveRDS(day10_ref, "data/interim/pid_day10_ref")
rm(encode_pid_day10_mxef)
# 
# # encode groud and day_mod_7 interaction effect
encode_group_day7_mxef <- glmmTMB(num_items_bought ~ (1|day_mod_7/group),
                             data=train_df, ziformula = ~1,
                             family= poisson)
group_day7_ref <- ranef(encode_group_day7_mxef)$cond$group
setDT(group_day7_ref, keep.rownames = TRUE)
names(group_day7_ref) <- c("group_day_mod_7", "group_day7_ref")
saveRDS(group_day7_ref, "data/interim/group_day7_inter_ref")
day7_ref <- ranef(encode_group_day7_mxef)$cond$day_mod_7
setDT(day7_ref, keep.rownames = TRUE)
names(day7_ref) <- c("day_mod_7", "day7_ref")
saveRDS(day7_ref, "data/interim/group_day7_ref")
rm(encode_group_day7_mxef)
# 
# # encode groud and day_mod_10 interaction effect
encode_group_day10_mxef <- glmmTMB(num_items_bought ~ (1|day_mod_10/group),
                                  data=train_df, ziformula = ~1,
                                  family= poisson)
group_day10_ref <- ranef(encode_group_day10_mxef)$cond$group
setDT(group_day10_ref, keep.rownames = TRUE)
names(group_day10_ref) <- c("group_day_mod_10", "group_day10_ref")
saveRDS(group_day10_ref, "data/interim/group_day10_ref")
day10_ref <- ranef(encode_group_day10_mxef)$cond$day_mod_10
setDT(day10_ref, keep.rownames = TRUE)
names(day10_ref) <- c("day_mod_10", "day10_ref")
saveRDS(day10_ref, "data/interim/group_day10_ref")
rm(encode_group_day10_mxef)

# encode category and day_mod_7 interaction effect
encode_category_day7_mxef <- glmmTMB(num_items_bought ~ (1|day_mod_7/category),
                                   data=train_df, ziformula = ~1,
                                   family= poisson)
category_day7_ref <- ranef(encode_category_day7_mxef)$cond$category
setDT(category_day7_ref, keep.rownames = TRUE)
names(category_day7_ref) <- c("category_day_mod_7", "category_day7_ref")
saveRDS(category_day7_ref, "data/interim/category_day7_inter_ref")
day7_ref <- ranef(encode_category_day7_mxef)$cond$day_mod_7
setDT(day7_ref, keep.rownames = TRUE)
names(day7_ref) <- c("category_mod_7", "day7_ref")
saveRDS(day7_ref, "data/interim/category_day7_ref")
rm(encode_category_day7_mxef)

# encode category and day_mod_10 interaction effect
encode_category_day10_mxef <- glmmTMB(num_items_bought ~ (1|day_mod_10/category),
                                     data=train_df, ziformula = ~1,
                                     family= poisson)
category_day10_ref <- ranef(encode_category_day10_mxef)$cond$category
setDT(category_day10_ref, keep.rownames = TRUE)
names(category_day10_ref) <- c("category_day_mod_10", "category_day10_ref")
saveRDS(category_day10_ref, "data/interim/category_day10_inter_ref")
day10_ref <- ranef(encode_category_day10_mxef)$cond$day_mod_10
setDT(day10_ref, keep.rownames = TRUE)
names(day10_ref) <- c("category_mod_10", "day10_ref")
saveRDS(day10_ref, "data/interim/category_day10_ref")
rm(encode_category_day10_mxef)

# encode content and day_mod_7 interaction effect
encode_content_day7_mxef <- glmmTMB(num_items_bought ~ (1|day_mod_7/content),
                                     data=train_df, ziformula = ~1,
                                     family= poisson)
content_day7_ref <- ranef(encode_content_day7_mxef)$cond$content
setDT(content_day7_ref, keep.rownames = TRUE)
names(content_day7_ref) <- c("content_day_mod_7", "content_day7_ref")
saveRDS(content_day7_ref, "data/interim/content_day7_inter_ref")
day7_ref <- ranef(encode_content_day7_mxef)$cond$day_mod_7
setDT(day7_ref, keep.rownames = TRUE)
names(day7_ref) <- c("content_mod_7", "day7_ref")
saveRDS(day7_ref, "data/interim/content_day7_ref")
rm(encode_content_day7_mxef)

# encode content and day_mod_10 interaction effect
encode_content_day10_mxef <- glmmTMB(num_items_bought ~ (1|day_mod_10/category),
                                      data=train_df, ziformula = ~1,
                                      family= poisson)
content_day10_ref <- ranef(encode_content_day10_mxef)$cond$content
setDT(content_day10_ref, keep.rownames = TRUE)
names(content_day10_ref) <- c("content_day_mod_10", "content_day10_ref")
saveRDS(content_day10_ref, "data/interim/content_day10_inter_ref")
day10_ref <- ranef(encode_content_day10_mxef)$cond$day_mod_10
setDT(day10_ref, keep.rownames = TRUE)
names(day10_ref) <- c("content_mod_10", "day10_ref")
saveRDS(day10_ref, "data/interim/content_day10_ref")
rm(encode_content_day10_mxef)

# ------------------------ Not using anymore ---------------------#
#We want to encode some high cardinality features:
# manufacturer, group, content, pharmForm, category

# fit a generalized mixed effect model with zero-inflated poisson distribution on 
# each categorical variable

# encode manufacturer effect
# encode_manufacturer_mxef <- glmmTMB(num_items_bought ~ (1|manufacturer), 
#                                     data = train_df, ziformula = ~1, 
#                                     family = poisson)
# manufacturer_ref <- ranef(encode_manufacturer_mxef)$cond$manufacturer
# setDT(manufacturer_ref, keep.rownames = TRUE)
# names(manufacturer_ref) <- c("manufacturer", "manufacturer_ref")
# saveRDS(manufacturer_ref, "data/processed/manufacturer_ref")
# rm(encode_manufacturer_mxef)
# 
# 
# # encode group
# encode_group_mxef <- glmmTMB(num_items_bought ~ (1|group), 
#                              data=train_df, ziformula = ~1, 
#                              family= poisson)
# group_ref <- ranef(encode_group_mxef)$cond$group
# setDT(group_ref, keep.rownames = TRUE)
# names(group_ref) <- c("group", "group_ref")
# saveRDS(group_ref, "data/processed/group_ref")
# rm(encode_group_mxef)
# 
# # encode content
# encode_content_mxef <- glmmTMB(num_items_bought ~ (1|content), 
#                                data=train_df, ziformula = ~1, 
#                                family= poisson)
# content_ref <- ranef(encode_content_mxef)$cond$content
# setDT(content_ref, keep.rownames = TRUE)
# names(content_ref) <- c("content", "content_ref")
# saveRDS(content_ref, "data/processed/content_ref")
# rm(encode_content_mxef)
# 
# # encode pharmForm
# encode_pharmForm_mxef <- glmmTMB(num_items_bought ~ (1|pharmForm), 
#                                  data=train_df, ziformula = ~1, 
#                                  family= poisson)
# pharmForm_ref <- ranef(encode_pharmForm_mxef)$cond$pharmForm
# setDT(pharmForm_ref, keep.rownames = TRUE)
# names(pharmForm_ref) <- c("pharmForm", "pharmForm_ref")
# saveRDS(pharmForm_ref, "data/processed/pharmForm_ref")
# rm(encode_pharmForm_mxef)
# 
# # encode category
# encode_category_mxef <- glmmTMB(num_items_bought ~ (1|category), 
#                                 data=train_df, ziformula = ~1, 
#                                 family= poisson)
# category_ref <- ranef(encode_category_mxef)$cond$category
# setDT(category_ref, keep.rownames = TRUE)
# names(category_ref) <- c("category", "category_ref")
# saveRDS(category_ref, "data/processed/category_ref")
# rm(encode_category_mxef)
# 
# # encode unit
# encode_unit_mxef <- glmmTMB(num_items_bought ~ (1|unit), 
#                                 data=train_df, ziformula = ~1, 
#                                 family= poisson)
# unit_ref <- ranef(encode_unit_mxef)$cond$unit
# setDT(unit_ref, keep.rownames = TRUE)
# names(unit_ref) <- c("unit", "unit_ref")
# saveRDS(unit_ref, "data/processed/unit_ref")
# rm(encode_unit_mxef)

