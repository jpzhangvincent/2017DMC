#!/usr/bin/env Rscript

library(data.table)
library(feather)


main = function() {
  message()
  DAYS = c(63, 77, 92)

  message("---------------------------------------- Train")
  lapply(DAYS, merge_set, "train")

  message("---------------------------------------- Test")
  lapply(DAYS, merge_set, "test")

  message("All done.\n")
}


merge_set = function(d, set) {
  # Load the relevant data set.
  IN = "../data/processed/end%i_%s.feather"

  data_path = sprintf(IN, d, set)
  df = data.table(read_feather(data_path))
  message(sprintf("Read: %s", data_path))
  setkey(df, lineID)

  # Merge every matching file in the prediction directory.
  PRED_PATTERN = "^end%i_%s" # preds files start `end##_SET_`
  PRED_DIR = "../data/preds1stLevel"

  re = sprintf(PRED_PATTERN, d, set)
  paths = list.files(PRED_DIR, re, full.names = TRUE)
  for (path in paths) {
    message(sprintf("  Merging: %s", path))
    pred = data.table(read_feather(path))
    setkey(pred, lineID)

    df = merge(df, pred, all.x = TRUE)
  }

  # Write
  OUT = "../data/layer2/end%i_%s_layer2.feather"

  out_path = sprintf(OUT, d, set)
  write_feather(df, out_path)
  message(sprintf("  WROTE: %s\n", out_path))

  invisible (NULL)
}


main()
