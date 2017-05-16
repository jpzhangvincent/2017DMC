#!/usr/bin/env Rscript
#
# This script does a sanity check on the processed data sets.
#

library(feather)


check = function(df, f) {
  counts = vapply(df, function(x) sum(f(x)), 1L)

  sort(counts[counts > 0])
}


paths = list.files("../data/processed", "feather$", full.names = TRUE)

lapply(paths, function(p) {
  print(p)
  df = read_feather(p)
  print(check(df, is.na))
  print(check(df, is.infinite))

  invisible (NULL)
})

message("Check finished.")
