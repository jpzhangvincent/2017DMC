#!/usr/bin/env Rscript
#
# Functions to read the DMC data.

library("readr")
library("tibble")


#' Read a DMC File
#'
read_dmc = function(file, ...) {
  read_delim(file, delim = "|", ...)
}


#' Write DMC Files to RDS
#'
#' @param data_dir (character) The data directory.
#' @param files (character) The data file names, without extensions.
write_dmc_rds = function(
  data_dir = "data",
  files = c("class", "items", "train")
) {
  files = file.path(data_dir, files)
  in_files = paste0(files, ".csv")
  out_files = paste0(files, ".rds")

  Map(function(i, o) {
    saveRDS(read_dmc(i), o)
  }, in_files, out_files)
}
