
## Minutes for 17/04/17

Some observations we might want to remove:
* availability == 4 (out of stock?) with order == 1
* (??) lines with PIDs that only appear once

competitorPrice, campaignIndex, pharmForm sometimes missing

What do other availability categories mean?
* More likely to be ordered for smaller values

Price usually lower than reference price.

Is there hidden information in pid, group?
* Run a clustering algorithm on the items data.
* Does group follow ##xx##xx bigram pattern?
* Is group ## hexadecimal?
* Price per unit range within groups?
* Some PIDs have identical attributes. Are they the same?
  + PID might correspond to URL for product page

Some PIDs in the test set are not in the training set.

How does adFlag interact with price?

For unit, "ST" is number of doses? What is "P"?

Average total basket price?

TODO:

* (Vincent) Clustering items
* Groups of PIDs...are they the same?
* Manufacturers
* German pharmacy abbreviations
* (Nick) Decoding the group
* (Hugo) PMI
* (Hugo) Unit unification

---

Feature engineering:

* one-hot encoding
* impact scores for categorical variables
* clustering PIDs (what about small # of obs?)
* adFlag and discount vs competitor
  + competitor price does not change much



## Minutes for 17/04/1 Meeting

Similar drugs might have an effect on each other; a user might click one drug
and then end up buying a different drug.

How many items were ordered per action?

How many different drugs per category and how many categories? Possibly a
relationship between equivalent drugs. "Popularity" effect.

Try cluster-based features. This may help with 

Estimate probability for purchase of a drug

Discrete model vs continuous model. Most of the 

First 60 days as training data; last 32 as test data.

Difference between price and competitor price might be a useful feature.

Cluster on item attributes.

Predicted clicks might be a useful feature.

Cumulative revenue for each product up to the current time point.

There might be items that have correlated sales if they are typically bought
together.

There might be stronger correlations between actions within days, if the
actions correspond to the same set of users.

Plans:

* __Haoran__ & __Shuhao__ suggested clustering product numbers based on product
  type. Similar drugs might have similar actions. Estimate number of orders per
  product type. We can estimate probability of sale for each product.
* __Hugo__ & __Jingyi__ suggested we develop a workflow for rapid testing of
  features and models. We should make this language agnostic (so the
  inputs/outputs should be CSV or similar).
