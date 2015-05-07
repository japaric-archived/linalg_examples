//! Multivariate linear regression using gradient descent
//!
//! Model:
//!
//! ```
//! y = x * theta + e
//!
//! y      Dependent variable     (scalar)
//! x      Independent variables  (1-by-n matrix)
//! theta  Parameters to estimate (n-by-1 matrix)
//! e      Error                  (scalar)
//! ```
//!
//! Cost function: Half of the mean squared error (MSE)
//!
//! ```
//! E = X * theta - Y
//! J = E' * E / 2 / m
//!
//! E      Error per observation          (m-by-1 matrix)
//! X      Observed independent variables (m-by-n matrix)
//! Y      Observed dependent variables   (m-by-1 matrix)
//! m      Number of observations         (integer)
//! theta  Parameters to estimate         (n-by-1 matrix)
//! ```
//!
//! Estimator: Gradient descent
//!
//! ```
//! loop
//!     E = X * theta - Y
//!     theta = theta - alpha / m * X' * E
//! until stop_condition
//!
//! E                (m-by-1 matrix)
//! X                (m-by-n matrix)
//! Y                (m-by-1 matrix)
//! alpha  Step size (scalar)
//! theta            (n-by-1 matrix)
//! ```

#![allow(non_snake_case)]
#![deny(warnings)]
#![feature(plugin)]
#![plugin(linalg_macros)]

extern crate cast;
extern crate env_logger;
extern crate linalg;
extern crate lines;
extern crate stats;
extern crate time;

#[macro_use]
extern crate log;

use std::fs::File;
use std::io::{BufReader, self};
use std::path::Path;

use cast::From as _0;
use linalg::prelude::*;
use linalg::{Col, ColMut, SubMat, SubMatMut, Transposed};
use lines::Lines;
use stats::univariate::Sample;

macro_rules! timeit {
    ($msg:expr, $e:expr) => {{
        let now = time::precise_time_ns();
        let out = $e;
        let elapsed = time::precise_time_ns() - now;
        println!(concat!($msg, " took {} ms"), f64::from_(elapsed) / 1_000_000.);
        out
    }}
}

fn main() {
    env_logger::init().unwrap();

    // Some dummy operation to force the initialization of OpenBLAS' runtime (~90 ms) here rather
    // than during the measurements below
    (&mat![1., 2.; 3., 4.].inv() * &mat![1., 2.; 3., 4.]).eval();

    let data = timeit!("Loading data", {
        load("mpg.tsv").unwrap()
    });

    // Number of observations
    let m = data.nrows();

    println!("{} observations", m);

    // Number of independent variables
    let n = data.ncols() - 1;

    println!("{} independent variables\n", n);

    let mut X = Mat::ones((m, n + 1));
    X[.., 1..] = data[.., 1..];
    let y = data.col(0);

    let (mu, sigma) = timeit!("Normalization", {
        normalize(&mut X[.., 1..])
    });

    println!("mean: {:?}", mu);
    println!("std deviation: {:?}\n", sigma);

    let ref mut theta = ColVec::zeros(n + 1);

    let alpha = 0.01;
    let max_niters = 100_000;

    let niters = timeit!("Gradient descent", {
        descent(&X, y, theta, alpha, max_niters)
    });

    println!("Estimated parameters: {:?}", theta);
    println!("Iterations required: {}", niters);
}

/// Evaluates the cost function for `theta`
///
/// X      (m, n)
/// y      (m, 1)
/// theta  (n, 1)
/// z      (m, 1)  Auxiliary buffer to avoid allocating
fn cost(X: &SubMat<f64>, y: &Col<f64>, theta: &Col<f64>, mut z: &mut Col<f64>) -> f64 {
    let m = f64::from_(X.nrows());

    z[..] = y - X * theta;

    let e = &*z;

    e.t() * e / 2. / m
}

/// Normalizes the independent variables
///
/// X  (m, n)
///
/// -> Returns a vector of means and a vector of standard deviations
fn normalize(X: &mut SubMat<f64>) -> (Vec<f64>, Vec<f64>) {
    let n = usize::from_(X.ncols());

    let mut mu = Vec::with_capacity(n);
    let mut sigma = Vec::with_capacity(n);

    for col in X.cols_mut() {
        let (mean, sd) = {
            let sample = Sample::new(col.as_slice().unwrap());
            let mean = sample.mean();

            (mean, sample.std_dev(Some(mean)))
        };

        mu.push(mean);
        sigma.push(sd);

        *col -= mean;
        *col /= sd;
    }

    (mu, sigma)
}

/// Performs the gradient descent algorithm to find the value of `theta` that minimizes the cost
/// function.
///
/// X           (m, n)
/// y           (m, 1)
/// theta       (n, 1)
/// alpha       scalar   Step size
/// max_niters  integer  Maximum number of iterations
///
/// -> Returns the number of iterations required to converge to a solution
fn descent(
    X: &SubMat<f64>,
    y: &Col<f64>,
    theta: &mut Col<f64>,
    alpha: f64,
    max_niters: u32,
) -> u32 {
    const TOL: f64 = 1e-5;

    let m = f64::from_(X.nrows());

    // Pre-allocate a column vector to avoid allocations in the loop
    let ref mut z = ColVec::zeros(X.nrows());

    let mut last_J = cost(X, y, theta, z);
    for i in 0..max_niters {
        // z = e = y - X * theta
        z[..] = y - X * theta;

        // theta = theta + alpha / m * x' * e
        *theta += alpha * X.t() * &*z / m;

        let J = cost(X, y, theta, z);

        debug!("i: {}, J: {}, theta: {:?}", i, J, theta);

        // Stop condition: `cost` reduced by less than `TOL`% in last iteration
        if (J - last_J).abs() / J.max(last_J) < TOL {
            return i
        }

        last_J = J;
    }

    max_niters
}

/// Loads data from a TSV file
fn load<P>(path: P) -> io::Result<Transposed<Mat<f64>>> where P: AsRef<Path> {
    fn load(path: &Path) -> io::Result<Transposed<Mat<f64>>> {
        let mut lines = Lines::from(BufReader::new(try!(File::open(path))));

        let mut v = vec![];

        let ncols = {
            let mut ncols = 0;

            for number in try!(lines.next().unwrap()).split_whitespace() {
                ncols += 1;
                v.push(number.parse().unwrap());
            }

            ncols
        };

        let mut nrows = 1;
        while let Some(line) = lines.next() {
            let line = try!(line);

            for number in line.split_whitespace() {
                v.push(number.parse().unwrap());
            }

            nrows += 1;
        }

        unsafe {
            Ok(Mat::from_raw_parts(v.into_boxed_slice(), (ncols, nrows)).t())
        }
    }

    load(path.as_ref())
}
