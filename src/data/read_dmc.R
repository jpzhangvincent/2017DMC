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


#' Write DMC Files to CSV
#'
#' @param data_dir (character) The data directory.
#' @param files (character) The data file names, without extensions.
write_dmc_csv = function(
  data_dir = "data/raw",
  files = c("class", "items", "train")
) {
  files = file.path(data_dir, files)
  in_files = paste0(files, ".txt")
  out_files = paste0(files, ".csv")

  Map(function(i, o) {
    write_csv(read_dmc(i), o)
  }, in_files, out_files)
}
