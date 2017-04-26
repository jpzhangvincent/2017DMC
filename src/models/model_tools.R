# EXAMPLE:
if (FALSE) { # NOTE: This example never runs.
  # SETUP ----------------------------------------
  set.seed(260)

  source("src/models/model_tools.R")
 
  dmc = dmc_read("data")
  dmc = dmc_merge(dmc) # merge train and items
  dmc = dmc_split(dmc) # split train into train and test on day 62

  attach(dmc)

  # MODEL LEVEL 1 (order) ----------------------------------------
  model1 = xgboost(data = data_xgb(train), label = train$order,
    max.depth = 2, eta = 1, nthread = 2, nround = 2,
    objective = "binary:logistic")
 
  preds1 = predict(model1, data_xgb(test))
  preds1 = ifelse(preds1 < 0.3, 0, 1)

  # MODEL LEVEL 2 (revenue) ----------------------------------------
  model2 = glm(quantity ~ ., family = poisson,
    data = with(train, cbind(data, quantity = quantity)[order == 1, ])
  )
  
  # FIXME: Predictions will fail since the test data contains novel levels for
  # the "group" variable.
  lambda = predict(model2, with(test, data[preds1 == 1, ]))
  preds2 = rpois(length(lambda), lambda)

  preds = preds1
  preds[preds1 == 1] = preds2

  # COMPUTE ERROR ----------------------------------------
  with(test, table(preds1, order))

  err = with(test, (preds * data$price - revenue)^2)

  detach(dmc)
} # END EXAMPLE


library("readr")
library("tibble")
library("xgboost")


#' Read DMC Data
#'
#' Read the 3 DMC data sets into a list with elements "train", "items", and
#' "test".
#'
#' @param data_dir The directory where the CSV files are located.
#'
dmc_read = function(data_dir = "../data") {
  paths = file.path(data_dir, c("train.csv", "items.csv", "class.csv"))

  f = function(file, ...) read_delim(file, delim = "|", ...)

  list(
    train = f(paths[[1]])
    , items = f(paths[[2]])
    , test = f(paths[[3]])
  )
}


#' Merge DMC Data
#'
#' @param dmc_list A list of DMC data frames, from `dmc_read()`.
#' @param drop A vector of column names to drop.
#'
dmc_merge = function(dmc_list, drop = c("pid", "lineID")) {
  # NOTE: Do not add columns that depend on multiple observations!
  dmc_list = within(dmc_list, {
    items$unit = tolower(items$unit)
    items$pharmForm = tolower(items$pharmForm)

    train$quantity = train$revenue / train$price
  })

  df = merge(dmc_list$train, dmc_list$items, by = "pid", all.x = TRUE,
    all.y = FALSE, sort = FALSE)

  df = df[order(df$lineID), ]

  # Remove columns that won't be used in the model.
  if (is.character(drop))
    df = df[!(names(df) %in% drop)]

  return (df)
}


#' Split DMC Data into Train & Test Sets
#'
#' Split a merged DMC data frame into train and test sets.
#'
#' This function returns a list with elements "train" and "test". Each of these
#' is a list with elements "data" and "labels".
#'
#' @param dmc A merged DMC data frame, from `dmc_merge()`.
#'
dmc_split = function(dmc, final_train_day = 62) {
  dmc = split(dmc,
    factor(dmc$day <= final_train_day, labels = c("test", "train"))
  )

  LABEL_COLS = c("click", "basket", "order", "revenue", "quantity")

  dmc = lapply(dmc, function(df) {
    is_label = names(df) %in% LABEL_COLS

    lst = unclass(df[is_label])
    lst$data = df[!is_label]
    return (lst)
  })

  return (dmc)
}


#' Prepare Data for xgboost
#'
#' This function converts a data frame into a numeric matrix. Categorical
#' variables are converted to their integer factor codes.
#'
data_xgb = function(x) data.matrix(as.data.frame(unclass(x$data)))
