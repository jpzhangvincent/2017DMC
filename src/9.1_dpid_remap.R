#!/usr/bin/env Rscript
#
# This script corrects the deduplicated_pid columns in the random effect
# encoding files.
#
# For example:
#  ranef  ->  correct
#  11099  ->     7323
#  16738  ->    16737
#   6557  ->     6556
#  11001  ->     5785
#  11442  ->    11439
#   4098  ->     4097
#  10227  ->    10226

library(feather)
library(stringr)

items = read_feather("../data/interim/1_clean_items.feather")

files = list.files("../data/merge/", "end.+ranef.rds", full.names = TRUE)

for (f in files) {
  rf = readRDS(f)
  message(sprintf("Reading: %s", f))
  rf$deduplicated_pid = as.integer(rf$deduplicated_pid)

  is_broken = which(!(rf$deduplicated_pid %in% items$deduplicated_pid))

  mapping = match(rf$deduplicated_pid[is_broken], items$pid)
  rf$deduplicated_pid[is_broken] = items$deduplicated_pid[mapping]

  if (!all(rf$deduplicated_pid %in% items$deduplicated_pid))
    stop("something went wrong!")

  f = str_replace(f, ".rds$", "_correct.rds")
  saveRDS(rf, f)
  message(sprintf("Wrote: %s", f))
}
