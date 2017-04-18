
* [ ] one-hot encoding
  * availability
  * manufacturer
* [ ] adFlag & (competitorPrice - price > 0)
* [ ] price vs rrp
  * discretize?
* [ ] competitorPrice vs rrp
* [ ] discretize day within month
  * by week
  * by 10-day period
* [ ] moving statistics for price per pid
  * [ ] mean
  * [ ] median
  * [ ] variance
* [ ] moving statistics for competitorPrice per pid
* [ ] day of week
* [ ] competitorPrice vs historical competitorPrice
  * [ ] max
  * [ ] previous
* [ ] price vs historical price
* [ ] content
  * [ ] multiply across "X"
  * [ ] split into two variables
* [ ] unify the units
  * separate variables for separate units
* [ ] manufacturer
  * [ ] number of PIDs
  * [ ] number of lines
  * [ ] value of products
  * [ ] vs adFlag
  * [ ] vs price per unit
* [ ] "rank" of PID within manufacturer
* [ ] price per unit
* [ ] rrp per unit
* [ ] PMI between current line and previous/next
* [ ] PID groups
  + might be useful for novel PIDs
* [ ] estimated probability of ordering
  * within PID
  * across all PIDs
  * incorporate noise / leave one out (?)
* [ ] lowercase pharmForm
  * possibly German abbreviations

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
