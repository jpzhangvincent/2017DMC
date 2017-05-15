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
import glob
import itertools as it
import numpy as np
import pandas as pd


def main():
    for path in glob.glob("../data/interim/3_*train.feather"):
        write_pmi(path, "group")


def write_pmi(path, by):
    print("Reading: %s" % path)
    train = feather.read_dataframe(path)
    train.pid = train.pid.astype(str)

    # Identify each line in the data set with a run.
    run_id = (train.order.shift(1) != train.order).cumsum()
    train = train.groupby(run_id)

    # Set up the path to the out directory.
    prefix = path.rsplit("_", 1)[0]
    prefix = prefix.replace("interim", "merge")
    prefix = prefix.replace("3_", "pmi_")

    by = "pid"
    runs = train.apply(lambda x: sorted(x[by].tolist()))
    runs = runs.tolist()

    del train

    pmi_pid = compute_pmi(runs, by)

    path = prefix + "_%s.feather" % by
    feather.write_dataframe(pmi_pid, path)
    print("Wrote: %s" % path)


def compute_pmi(runs, by):
    pair_counts = co.Counter(c for r in runs for c in it.combinations(r, 2))
    counts = co.Counter(x for r in runs for x in r)

    # Compute pointwise mutual information (PMI).
    pmi = pd.DataFrame({
            "group": x + "/" + y,
            "pmi_" + by: np.log(pair_counts[(x, y)] / counts[x] * counts[y])
        } for x, y in pair_counts.keys())

    return pmi


if __name__ == "__main__":
    main()
