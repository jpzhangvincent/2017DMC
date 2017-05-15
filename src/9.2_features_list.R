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

# Default to all features categorical, unless they are numeric.
features$type = "categorical"
features$type[features$rclass == "numeric"] = "numeric"

# List labels here.
LABELS = c("basket", "click", "order", "revenue", "order_qty")
features$type[features$name %in% LABELS] = "label"

# Override numeric features here.
NUMERIC = c("day")
features$type[features$name %in% NUMERIC] = "numeric"

# Override categorical features here.
CATEGORICAL = c()
features$type[features$name %in% CATEGORICAL] = "categorical"


ord = with(features, order(type, name))
features = features[ord, c("name", "type", "rclass")]

path = "../data/feature_list.csv"
write_csv(features, path)
message(sprintf("Wrote: %s", path))

