#!/bin/bash

URL=https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original

curl $URL \
    | grep -v NA \
    | cut -d '"' -f 1 \
    > mpg.tsv
