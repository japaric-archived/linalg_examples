# Normal equation

This crate solves a multiclass classification problem using logistic regression and the one vs rest
strategy.

For reference a Python implementation is also provided.

You can run this example with the following commands

```
$ ./fetch_data.sh
$ less mpg.tsv
$ cargo run [--release]
$ python normal.py
```

# Data

The data used for this example can be found [here]

[here]: https://archive.ics.uci.edu/ml/datasets/Iris

# Line count

## Rust

```
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Rust                             1             46             59            174
-------------------------------------------------------------------------------
```

## Python

```
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                           1             33              1             66
-------------------------------------------------------------------------------
```

# Execution time

## Rust (debug profile)

```
Loading data took 6.643995 ms

Setosa vs rest took 5500.947223 ms
Estimated parameters: Col([0.49901, 0.786968, 2.749511, -4.280391, -1.96856])
Iterations required: 89815

Versicolor vs rest took 318.718091 ms
Estimated parameters: Col([0.409998, 0.408378, -1.347258, 0.428646, -0.946081])
Iterations required: 5028

Virginica vs rest took 1822.947782 ms
Estimated parameters: Col([-2.160549, -2.964011, -2.786403, 4.321836, 4.517111])
Iterations required: 28931
```

## Rust (release profile)

```
Loading data took 0.288227 ms

Setosa vs rest took 763.643324 ms
Estimated parameters: Col([0.49901, 0.786968, 2.749511, -4.280391, -1.96856])
Iterations required: 89815

Versicolor vs rest took 47.032565 ms
Estimated parameters: Col([0.409998, 0.408378, -1.347258, 0.428646, -0.946081])
Iterations required: 5028

Virginica vs rest took 257.838009 ms
Estimated parameters: Col([-2.160549, -2.964011, -2.786403, 4.321836, 4.517111])
Iterations required: 28931
```

## Python

```
Loading data took 0.72193145752 ms

Setosa vs rest took 14813.792944 ms
Estimated parameters: [ 0.4990102   0.78696804  2.74951088 -4.28039079 -1.96855985]
Iterations required: 89815

Versicolor vs rest took 805.121898651 ms
Estimated parameters: [ 0.40999834  0.40837752 -1.34725773  0.42864567 -0.94608068]
Iterations required: 5028

Virginica vs rest took 4585.66999435 ms
Estimated parameters: [-2.16054907 -2.96401082 -2.78640311  4.32183636  4.51711112]
Iterations required: 28931
```

# Memory usage

## Rust

(There was no difference between the debug and the optimized builds)

```
==31325== HEAP SUMMARY:
==31325==     in use at exit: 5,392 bytes in 12 blocks
==31325==   total heap usage: 109 allocs, 97 frees, 184,133 bytes allocated
==31325==
==31325== LEAK SUMMARY:
==31325==    definitely lost: 0 bytes in 0 blocks
==31325==    indirectly lost: 0 bytes in 0 blocks
==31325==      possibly lost: 2,128 bytes in 7 blocks
==31325==    still reachable: 3,264 bytes in 5 blocks
==31325==         suppressed: 0 bytes in 0 blocks
```

## Python

```
==31697==     in use at exit: 2,891,230 bytes in 3,792 blocks
==31697==   total heap usage: 4,352,142 allocs, 4,348,350 frees, 5,169,995,676 bytes allocated
==31697==
==31697== LEAK SUMMARY:
==31697==    definitely lost: 304 bytes in 1 blocks
==31697==    indirectly lost: 0 bytes in 0 blocks
==31697==      possibly lost: 154,938 bytes in 91 blocks
==31697==    still reachable: 2,735,988 bytes in 3,700 blocks
==31697==         suppressed: 0 bytes in 0 blocks
```
