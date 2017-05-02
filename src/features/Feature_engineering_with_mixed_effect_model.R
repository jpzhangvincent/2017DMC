library(data.table)
library(glmmTMB) 

# Reasons for using generalized mixed effect model
# - high dimensionality of categorical variable, so it's better to 
#   infer and quantify the impacts of different levels 
# - noval levels in test set, so treating the variable as random effect sounds intuitive
# but in the end, we would rely on the feature election to verify this approach 

orig_train_df <- readRDS("notebooks/train_cleanformat.rds")
train_df <- orig_train_df[day<=62]
#valid_df <- orig_train_df[day>62]

string_vars <- c("pid", "lineID", "manufacturer", "group", "content", "unit", "pharmForm", 
                 "genericProduct", "salesIndex", "category", "campaignIndex")
train_df[, lapply(.SD, function(x) length(unique(x))),.SDcols = string_vars]

#We want to encode some high cardinality features:
# manufacturer, group, content, pharmForm, category

# fit a generalized mixed effect model with zero-inflated poisson distribution on 
# each categorical variable

# encode manufacturer effect
encode_manufacturer_mxef <- glmmTMB(num_items_bought ~ (1|manufacturer), 
                                    data = train_df, ziformula = ~1, 
                                    family = poisson)
manufacturer_ref <- ranef(encode_manufacturer_mxef)$cond$manufacturer
setDT(manufacturer_ref, keep.rownames = TRUE)
names(manufacturer_ref) <- c("manufacturer", "manufacturer_ref")
saveRDS(manufacturer_ref, "data/processed/manufacturer_ref")
rm(encode_manufacturer_mxef)


# encode group
encode_group_mxef <- glmmTMB(num_items_bought ~ (1|group), 
                             data=train_df, ziformula = ~1, 
                             family= poisson)
group_ref <- ranef(encode_group_mxef)$cond$group
setDT(group_ref, keep.rownames = TRUE)
names(group_ref) <- c("group", "group_ref")
saveRDS(group_ref, "data/processed/group_ref")
rm(encode_group_mxef)

# encode content
encode_content_mxef <- glmmTMB(num_items_bought ~ (1|content), 
                               data=train_df, ziformula = ~1, 
                               family= poisson)
content_ref <- ranef(encode_content_mxef)$cond$content
setDT(content_ref, keep.rownames = TRUE)
names(content_ref) <- c("content", "content_ref")
saveRDS(content_ref, "data/processed/content_ref")
rm(encode_content_mxef)

# encode pharmForm
encode_pharmForm_mxef <- glmmTMB(num_items_bought ~ (1|pharmForm), 
                                 data=train_df, ziformula = ~1, 
                                 family= poisson)
pharmForm_ref <- ranef(encode_pharmForm_mxef)$cond$pharmForm
setDT(pharmForm_ref, keep.rownames = TRUE)
names(pharmForm_ref) <- c("pharmForm", "pharmForm_ref")
saveRDS(pharmForm_ref, "data/processed/pharmForm_ref")
rm(encode_pharmForm_mxef)

# encode category
encode_category_mxef <- glmmTMB(num_items_bought ~ (1|category), 
                                data=train_df, ziformula = ~1, 
                                family= poisson)
category_ref <- ranef(encode_category_mxef)$cond$category
setDT(category_ref, keep.rownames = TRUE)
names(category_ref) <- c("category", "category_ref")
saveRDS(category_ref, "data/processed/category_ref")
rm(encode_category_mxef)

# encode unit
encode_unit_mxef <- glmmTMB(num_items_bought ~ (1|unit), 
                                data=train_df, ziformula = ~1, 
                                family= poisson)
unit_ref <- ranef(encode_unit_mxef)$cond$unit
setDT(unit_ref, keep.rownames = TRUE)
names(unit_ref) <- c("unit", "unit_ref")
saveRDS(unit_ref, "data/processed/unit_ref")
rm(encode_unit_mxef)

# encode pid
encode_pid_mxef <- glmmTMB(num_items_bought ~ (1|pid), 
                           data=train_df, ziformula = ~1, 
                           family= poisson)
pid_ref <- ranef(encode_pid_mxef)$cond$pid
setDT(pid_ref, keep.rownames = TRUE)
names(pid_ref) <- c("pid", "pid_ref")
saveRDS(pid_ref, "data/processed/pid_ref")
rm(encode_pid_mxef)