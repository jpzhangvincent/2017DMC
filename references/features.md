# Planned Features

Output variables in standard CSV files:
  * Include `lineid` for variables on `train.csv` / `class.csv`
  * Include `pid` for variables on `items.csv`
  * Filename begins with the dataset name (`train`, `class`, or `items`) and
    ends with `.csv`

When naming files and variables:
  * Use underscores (not spaces, dashes, etc)
  * Use lowercase
  * Be as explicit as possible (length doesn't matter)
  * Add the name to `variable_names.md` file in `references/` directory

## Shuhao
* [ ] presentation slides (by Sunday morning)
  * introduce data (topic, variables)
  * exploratory results
  * plans for feature engineering
  * potential models (but ask class for advice/feedback)

## Nick
* [x] price per unit
* [ ] day of week
* [ ] PID groups
  + might be useful for novel PIDs
* [x] lowercase pharmForm
  * [ ] group by German abbreviations


## Jingyi
* [ ] adFlag & (competitorPrice - price > 0)
* [ ] price vs rrp
  * discretize?
* [ ] competitorPrice vs rrp

## Haoran
* [ ] moving statistics for price per pid
  * [ ] mean
  * [ ] median
  * [ ] variance
* [ ] moving statistics for competitorPrice per pid
* [ ] rrp per unit

## Lingfei
* [ ] competitorPrice vs historical competitorPrice
  * [ ] max
  * [ ] previous
* [ ] price vs historical price
  * windowed difference?

## Vincent
* [ ] content
  * [x] multiply across "X"
  * [ ] split into two variables
* [ ] "rank" of PID within manufacturer
* [x] vtreat / one-hot encoding / impact encoding
  * availability
  * manufacturer

## Olivia
* manufacturer
  * [ ] number of PIDs
  * [ ] number of lines
  * [ ] value of products
  * [ ] vs adFlag
  * [ ] vs price per unit

## Weitong
* [ ] deduplicate PIDs
  * [ ] 
* [ ] estimated probability of ordering
  * within PID
  * across all PIDs
  * incorporate noise / leave one out (?)



## Hugo
* [x] PMI between current line and previous/next
* [x] unify the units
  * separate variables for separate units
* [ ] discretize day within month
  * by week
  * by 10-day period

## train.csv
lineID
day
pid
adFlag
availability
competitorPrice

click
basket
order
price
revenue

## items.csv

pid
manufacturer
group
content
unit
pharmForm
genericProduct
salesIndex
category
campaignIndex
rrp
