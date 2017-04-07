
## Minutes for 17/04/17 Meeting

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
