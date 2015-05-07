# Normal equation

This crate performs a multivariate linear regression by solving the normal equation.

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

[here]: https://archive.ics.uci.edu/ml/datasets/Auto+MPG

# With operator sugar

`main.ng.rs` is how I expect the program to look if/when all [these changes] land in the compiler.
I recommend you look at the diff:

[these changes]: https://github.com/japaric/linalg.rs#improving-operator-sugar

```
$ [color]diff -u src/main.rs src/main.ng.rs | less -r
```

# Line count

## Rust

```
$ cloc src/main.rs
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Rust                             1             28             28             72
-------------------------------------------------------------------------------
```

## Python

```
$ cloc normal.py
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                           1             13              1             35
-------------------------------------------------------------------------------
```

# Execution time

Given that both `NumPy` and `linalg` offload their work to BLAS routines, their execution times in
this example are similar.

## Rust (debug profile)

```
$ cargo run
Loading data took 10.147551 ms
392 observations
7 independent variables

Solving the normal equation took 9.264773 ms
Estimated parameters: Col([-17.218435, -0.493376, 0.019896, -0.016951, -0.006474, 0.080576, 0.750773, 1.42614])
```

## Rust (release profile)

```
$ cargo run --release
Loading data took 0.749544 ms
392 observations
7 independent variables

Solving the normal equation took 5.903862 ms
Estimated parameters: Col([-17.218435, -0.493376, 0.019896, -0.016951, -0.006474, 0.080576, 0.750773, 1.42614])
```

## Python

```
$ python normal.py
Loading data took 3.00312042236 ms
392 observations
7 independent variables

Solving the normal equation took 7.46488571167 ms
Estimated parameters: [ -1.72184346e+01  -4.93376319e-01   1.98956437e-02  -1.69511442e-02
  -6.47404340e-03   8.05758383e-02   7.50772678e-01   1.42614050e+00]
```

# Memory usage

## Rust

(There was no difference between the debug and the optimized builds)

```
$ valgrind target/debug/normal
==32706== HEAP SUMMARY:
==32706==     in use at exit: 5,392 bytes in 12 blocks
==32706==   total heap usage: 127 allocs, 115 frees, 273,933 bytes allocated
==32706==
==32706== LEAK SUMMARY:
==32706==    definitely lost: 0 bytes in 0 blocks
==32706==    indirectly lost: 0 bytes in 0 blocks
==32706==      possibly lost: 2,128 bytes in 7 blocks
==32706==    still reachable: 3,264 bytes in 5 blocks
==32706==         suppressed: 0 bytes in 0 blocks
```

## Python

```
$ valgrind python normal.py
==32558== HEAP SUMMARY:
==32558==     in use at exit: 2,888,241 bytes in 3,775 blocks
==32558==   total heap usage: 19,232 allocs, 15,457 frees, 19,023,251 bytes allocated
==32558==
==32558== LEAK SUMMARY:
==32558==    definitely lost: 304 bytes in 1 blocks
==32558==    indirectly lost: 0 bytes in 0 blocks
==32558==      possibly lost: 154,938 bytes in 91 blocks
==32558==    still reachable: 2,732,999 bytes in 3,683 blocks
==32558==         suppressed: 0 bytes in 0 blocks
```
