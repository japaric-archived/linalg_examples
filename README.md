# `linalg` examples

The main goal of these examples is to showcase `linalg` API. A secondary goal is to compare the
ergonomics of doing numerical computing in Rust vs in other languages, to this extend
implementations of each example are provided in other languages. And, as this library currently
lacks (multi-language) benchmarks, some non-scientific measurements are included (I mainly wanted
to check that `linalg` is not slower than NumPy).

Examples included:

- `descent`: Multivariate [linear regression] using the iterative [gradient descent] algorithm
- `normal`: Multivariate linear regression using the [normal equation]

[linear regression]: https://en.wikipedia.org/wiki/Linear_regression
[gradient descent]: https://en.wikipedia.org/wiki/Gradient_descent
[normal equation]: https://en.wikipedia.org/wiki/Ordinary_least_squares#Estimation

Each example is a cargo project.

## Meta

Information about the system where the measurements were performed

### Rust

```
$ rustc -Vv
rustc 1.1.0-nightly (c42c1e7a6 2015-05-02) (built 2015-05-02)
binary: rustc
commit-hash: c42c1e7a678a27bb63c24fb1eca2866af4e3ab7a
commit-date: 2015-05-02
build-date: 2015-05-02
host: x86_64-unknown-linux-gnu
release: 1.1.0-nightly
```

### Python

```
$ python -V
Python 2.7.9
```

### Kernel

```
Linux ideapad 4.0.1 #1 SMP PREEMPT Wed Apr 29 13:47:28 PET 2015 x86_64 GNU/Linux
```

### CPU

```
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                8
On-line CPU(s) list:   0-7
Thread(s) per core:    2
Core(s) per socket:    4
Socket(s):             1
NUMA node(s):          1
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 60
Model name:            Intel(R) Core(TM) i7-4702MQ CPU @ 2.20GHz
Stepping:              3
CPU MHz:               801.453
CPU max MHz:           3200.0000
CPU min MHz:           800.0000
BogoMIPS:              4389.99
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              6144K
NUMA node0 CPU(s):     0-7
```

## License

These examples are dual licensed under the Apache 2.0 license and the MIT license.

See LICENSE-APACHE and LICENSE-MIT for more details.
