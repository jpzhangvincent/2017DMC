library(data.table)
library(stringr)
library(h2o)

# Data Preparation
item_df <- fread('data/raw/items.csv', na.strings = c('', ' ', 'NA'))
# Stringify --------------------
str_cols <- c("manufacturer", "group", "content", "unit", "pharmForm",
              "salesIndex", "category", "campaignIndex")
item_df[, (str_cols) := lapply(.SD, str_to_lower), .SDcols = str_cols]
item_df[, deduplicated_pid := pid[[1]], by = setdiff(colnames(item_df), "pid")]

item_df[, `:=`(
  pharmForm = ifelse(is.na(pharmForm), "?", pharmForm) , 
  category = ifelse(is.na(category), "?", category),
  campaignIndex = ifelse(is.na(campaignIndex), "?", campaignIndex)
)]
item_df[, lapply(.SD, function(x) sum(is.na(x)==T))]

manufacturer_ref <- readRDS("data/interim/train63d_manufacturer_ref")
group_ref <- readRDS("data/interim/train63d_group_ref")
pharmForm_ref <- readRDS("data/interim/train63d_pharmForm_ref")
content_ref <- readRDS("data/interim/train63d_content_ref")
unit_ref <- readRDS("data/interim/train63d_unit_ref")
category_ref <- readRDS("data/interim/train63d_category_ref")

item_df <- merge(item_df, manufacturer_ref, by = "manufacturer", all.x=T)
item_df <- merge(item_df, group_ref, by = "group", all.x=T)
item_df <- merge(item_df, content_ref, by = "content", all.x=T)
item_df <- merge(item_df, pharmForm_ref, by = "pharmForm", all.x=T)
item_df <- merge(item_df, unit_ref, by = "unit", all.x=T)
item_df <- merge(item_df, category_ref, by = "category", all.x=T)

to_fix <- c("manufacturer_ref", "content_ref", "pharmForm_ref", 
            "category_ref")
item_df[, (to_fix) := lapply(.SD, function(x) ifelse(is.na(x)==T, 
                                                     mean(x,na.rm = T), x)), .SDcols = to_fix]


# Clustering 
# k-modes algorithm in h2o
# Reference: 

h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "5G")  #max mem size is the maximum memory to allocate to H2O
items_cluster.hex <- as.h2o(item_df, destination_frame = "item.hex")
items_cluster.hex$salesIndex <- as.factor(items_cluster.hex$salesIndex)
items_cluster.hex$campaignIndex <- as.factor(items_cluster.hex$campaignIndex)
items_cluster.hex$genericProduct <- as.factor(items_cluster.hex$genericProduct)


cluster_vars <- c("category_ref", "group_ref", "pharmForm_ref", "content_ref", 
                  "manufacturer_ref", "content_ref", "unit_ref", "rrp", 
                  "genericProduct", "salesIndex", "campaignIndex")

item_km <- h2o.kmeans(training_frame = items_cluster.hex, k = 30, 
           x = cluster_vars, max_iterations = 100, seed = 1234 )
summary(item_km)
cluster_preds <- as.data.frame(h2o.predict(item_km, newdata = items_cluster.hex))
names(cluster_preds) <- "cluster"
cluster_info <- cbind(deduplicated_pid = item_df$deduplicated_pid, cluster_preds)
saveRDS(cluster_info, "data/interim/item_cluster_feature.Rds")
