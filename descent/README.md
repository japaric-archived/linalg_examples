# Gradient descent

This crate performs a multivariate linear regression using the iterative gradient descent algorithm

For reference a Python implementation is also provided.

You can run this example with the following commands

```
$ ./fetch_data.sh
$ less mpg.tsv
$ cargo run [--release]
$ python descent.py
```

# Data

The data used for this example can be found [here]

[here]: https://archive.ics.uci.edu/ml/datasets/Auto+MPG

# Line count

## Rust

```
$ cloc src/main.rs
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Rust                             1             53             74            130
-------------------------------------------------------------------------------
```

## Python

```
$ cloc descent.py
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                           1             22              1             52
-------------------------------------------------------------------------------
```

# Execution time

Although `Numpy` and `linalg` call the same BLAS routines, the Rust version has the edge here
because it avoids allocating in the iterative gradient descent routine. The difference in execution
time is around one order of magnitude for this workload, but will vary depending on the number of
iterations performed.

## Rust (debug profile)

```
$ cargo run
Loading data took 10.195174 ms
392 observations
7 independent variables

Normalization took 0.337333 ms
mean: [5.471939, 194.41199, 104.469388, 2977.584184, 15.541327, 75.979592, 1.576531]
std deviation: [1.705783, 104.644004, 38.49116, 849.40256, 2.758864, 3.683737, 0.805518]

Gradient descent took 10.100251 ms
Estimated parameters: Col([23.445918, -0.381446, 0.708861, -0.878851, -4.548997, -0.032518, 2.698499, 1.084316])
Iterations required: 2402
```

## Rust (release profile)

```
$ cargo run --release
Loading data took 0.751526 ms
392 observations
7 independent variables

Normalization took 0.027174 ms
mean: [5.471939, 194.41199, 104.469388, 2977.584184, 15.541327, 75.979592, 1.576531]
std deviation: [1.705783, 104.644004, 38.49116, 849.40256, 2.758864, 3.683737, 0.805518]

Gradient descent took 6.047204 ms
Estimated parameters: Col([23.445918, -0.381446, 0.708861, -0.878851, -4.548997, -0.032518, 2.698499, 1.084316])
Iterations required: 2402
```

## Python

```
$ python descent.py
Loading data took 3.00288200378 ms
392 observations
7 independent variables

Normalization took 0.149011611938 ms
mean: [  5.47193878e+00   1.94411990e+02   1.04469388e+02   2.97758418e+03
   1.55413265e+01   7.59795918e+01   1.57653061e+00]
std deviation: [  1.70360611e+00   1.04510444e+02   3.84420327e+01   8.48318447e+02
   2.75534291e+00   3.67903490e+00   8.04490081e-01]

Gradient descent took 70.1060295105 ms
Estimated parameters: [ 23.44591837  -0.38111383   0.70949687  -0.87722253  -4.54478318
  -0.03205276   2.69516261   1.08298208]
Iterations required: 2399
```

# Memory usage

## Rust

(There was no different between the debug and the optimized builds)

```
$ valgrind target/debug/descent
==29449== HEAP SUMMARY:
==29449==     in use at exit: 5,392 bytes in 12 blocks
==29449==   total heap usage: 97 allocs, 85 frees, 203,629 bytes allocated
==29449==
==29449== LEAK SUMMARY:
==29449==    definitely lost: 0 bytes in 0 blocks
==29449==    indirectly lost: 0 bytes in 0 blocks
==29449==      possibly lost: 2,128 bytes in 7 blocks
==29449==    still reachable: 3,264 bytes in 5 blocks
==29449==         suppressed: 0 bytes in 0 blocks
```

## Python

```
$ valgrind python descent.py
==29611== HEAP SUMMARY:
==29611==     in use at exit: 2,887,037 bytes in 3,785 blocks
==29611==   total heap usage: 55,413 allocs, 51,628 frees, 106,249,808 bytes allocated
==29611==
==29611== LEAK SUMMARY:
==29611==    definitely lost: 304 bytes in 1 blocks
==29611==    indirectly lost: 0 bytes in 0 blocks
==29611==      possibly lost: 154,938 bytes in 91 blocks
==29611==    still reachable: 2,731,795 bytes in 3,693 blocks
==29611==         suppressed: 0 bytes in 0 blocks
```
