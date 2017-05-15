#!/usr/bin/env Rscript
#
# This script outputs a list of features.
#

library(feather)
library(readr)


tr = read_feather("../data/processed/end63_train.feather")

features = vapply(tr, class, "")
features = data.frame(name = names(features), rclass = features)
rownames(features) = NULL

features$type = "discrete"
features$type[features$rclass == "numeric"] = "discrete"
LABELS = c("basket", "click", "order", "revenue", "order_qty")
features$type[features$name %in% LABELS] = "label"

ord = with(features, order(type, name))
features = features[ord, c("name", "type", "rclass")]

path = "../data/feature_list.csv"
write_csv(features, path)
message(sprintf("Wrote: %s", path))

