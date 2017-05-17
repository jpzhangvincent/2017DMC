#!/usr/bin/env Rscript

library(feather)

to_fix = list.files("../data/preds1stLevel", "nn.feather$", full.names = TRUE)
FIX_COLS = c("lineID", "TRUE.")

for (f in to_fix) {
  df = read_feather(f)
  if (!all(FIX_COLS %in% colnames(df)))
    stop(sprintf("ERROR: %s", f))

  df = df[FIX_COLS]
  colnames(df) = c("lineID", "preds_nn")

  write_feather(df, f)
  message(sprintf("Wrote: %s", f))
}
