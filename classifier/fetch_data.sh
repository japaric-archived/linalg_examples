#!/bin/bash

URL=https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data

curl $URL \
    > iris.csv
