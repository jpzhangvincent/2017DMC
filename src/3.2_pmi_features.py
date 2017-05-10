#!/usr/bin/env python3
#
# This script computes the pointwise mutual information (PMI) for several
# different features.
#
# The PMI measures how likely it is that a pair of levels will co-occur (in
# this case, in the same order run). Large PMI values indicate the two levels
# are more likely to co-occur than to occur separately.
#

import collections as co
import feather
import itertools as it
import numpy as np
import pandas as pd


def main():
    train = feather.read_dataframe("../data/interim/3_end63_train.feather")
    train.pid = train.pid.astype(str)

    # Identify each line in the data set with a run.
    run_id = (train.order.shift(1) != train.order).cumsum()
    train = train.groupby(run_id)

    pmi_pid = compute_pmi(train, "pid")
    feather.write_dataframe(pmi_pid, "../data/merge/pmi_pid.feather")
    print("Wrote: ../data/merge/pmi_pid.feather")
    
    pmi_group = compute_pmi(train, "group")
    feather.write_dataframe(pmi_group, "../data/merge/pmi_group.feather")
    print("Wrote: ../data/merge/pmi_group.feather")


def compute_pmi(runs, group):
    runs = runs.apply(lambda x: sorted(x[group].tolist()))
    runs = runs.tolist()

    pair_counts = co.Counter(c for r in runs for c in it.combinations(r, 2))
    counts = co.Counter(x for r in runs for x in r)

    # Compute pointwise mutual information (PMI).
    pmi = pd.DataFrame({
            "group": x + "/" + y,
            "pmi_" + group: np.log(pair_counts[(x, y)] / counts[x] * counts[y])
        } for x, y in pair_counts.keys())

    return pmi


if __name__ == "__main__":
    main()
