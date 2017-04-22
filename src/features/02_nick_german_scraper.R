#!/usr/bin/env Rscript

source("src/data/read_dmc.R")

library(readr)
library(rvest)
library(stringr)
library(xml2)


merge_desc = function() {
  # Setup
  items = read_dmc("data/items.csv")
  items_out = items["pid"]

  lc_pharm = tolower(items$pharmForm)

  # Scrape and merge
  kp = scrape_kohlpharma()
  i = match(lc_pharm, kp$abbrev)
  items_out$kp_desc = kp$kp_desc[i]

  dm = scrape_docmorris()
  j = match(lc_pharm, dm$abbrev1)
  is_na = is.na(j)
  j[is_na] = match(lc_pharm, dm$abbrev2)[is_na]
  is_na = is.na(j)
  j[is_na] = match(lc_pharm, dm$abbrev3)[is_na]

  items_out$dm_desc = dm$dm_desc[j]
  items_out$dm_group = dm$dm_group[j]

  return (items_out)
}


scrape_kohlpharma = function(f = "data/external/Kohlpharma.html") {
  html = read_html(f)
  tab = html_table(html)[[1]]

  colnames(tab) = c("abbrev", "kp_desc")
  tab = as_data_frame(lapply(tab, tolower))

  return (tab)
}


scrape_docmorris = function(f = "data/external/DocMorris-Blog.html") {
  html = read_html(f)
  content = xml_find_first(html, "//div[contains(@class, 'niceText')]")

  headers = xml_find_all(content, "./h2|./h3")
  headers = xml_text(headers)

  tables = xml_find_all(content, "table")
  tables = Map(function(tab, group) {
    tab = html_table(tab)
    tab = as_data_frame(tab)
    colnames(tab) = c("abbrev", "dm_desc")

    tab$dm_group = group

    return (tab)
  }, tables, headers)
  tab = do.call(rbind, tables)

  tab$dm_desc = tolower(tab$dm_desc)
  tab$dm_group = tolower(tab$dm_group)

  tab$abbrev = tolower(tab$abbrev)
  abbrevs = str_split_fixed(tab$abbrev, "od[.]", 3)
  tab$abbrev1 = str_trim(abbrevs[, 1])
  tab$abbrev2 = str_trim(abbrevs[, 2])
  tab$abbrev3 = str_trim(abbrevs[, 3])

  return (tab)
}


items_out = merge_desc()
write_csv(items_out, "data/interim/items_german_pharm_form.csv")
