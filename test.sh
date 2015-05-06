#!/bin/bash

set -e
set -x

for example in */; do
    pushd $example
    ./fetch_data.sh
    cargo run
    popd
done
