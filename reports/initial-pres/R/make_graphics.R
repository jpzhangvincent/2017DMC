#!/usr/bin/env Rscript

library("ggplot2")
library("readr")
library("tibble")

read_dmc = function(file, ...) {
  read_delim(file, delim = "|", ...)
}

train = read_dmc("../data/train.csv")
items = read_dmc("../data/items.csv")


# Runs Plot ----------------------------------------
train$Quantity = train$revenue / train$price

plt = ggplot(train[1:100, ], aes(x = lineID, y = Quantity)) + geom_point() +
  geom_line() + labs(title = "Runs")

ggsave("runs.png", plt, "png", "../graphics", width = 4, height = 4)


# Run Length Distributions ----------------------------------------

# Find the start and end points of each run.
runs = cumsum(head(train$order, -1) != tail(train$order, -1))
runs = c(0, runs)
#rbind(runs, df$order)

# How long are the runs?
lengths = table(runs)

run_type = as.numeric(names(lengths)) %% 2
run_df = data.frame(length = c(lengths), order = run_type)


plt = ggplot(run_df,
    aes(length, fill = factor(order, labels = c("No Order", "Order")))
  ) +
  geom_bar(position = "dodge") + xlim(0, 50) +
  ggtitle("Run Length Distribution") + xlab("Run Length") + ylab("Frequency") +
  guides(fill = guide_legend(title = "Run Type", direction = "horizontal")) +
  theme(legend.position = "bottom")

ggsave("run_length.png", plt, "png", "../graphics", width = 4, height = 4)

