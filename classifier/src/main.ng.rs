//! Multiclass classification using logistic regression and one vs rest strategy
//!
//! Model: Logistic regression (for binary classification)
//!
//! ```
//! h = S(x * theta)
//! S(z) = 1 / (1 + e^-z)
//!
//! S:     Sigmoid function
//! e      Euler's number for exponentiation                        (scalar)
//! h:     Probability that subject belong to the "positive" class  (scalar)
//! n      Number of features                                       (integer)
//! theta  Parameters to estimate (1-by-n matrix)                   (1-by-n matrix)
//! x      Subject's features                                       (n-by-1 matrix)
//! ```
//!
//! Cost function `J`
//!
//! ```
//! H = S(X * theta)
//! J = (-Y * log(H) - (1 - Y) * log(1 - H)) / m
//! grad = (X' * (X * theta - Y)) / m
//!
//! H     Estimated probabilities        (m-by-1 matrix)
//! X     Observed features              (m-by-n matrix)
//! Y     Observed classification        (m-by-1 matrix)
//! grad  Gradient of the cost function  (n-by-1 matrix)
//! m     Number of observations         (integer)
//! theta Parameters to estimate         (n-by-1 matrix)
//! ```
//!
//! Solver: Gradient descent
//!
//! loop
//!     theta = theta - alpha * grad(theta)
//! until stop condition
//!
//! alpha Step size                     (scalar)
//! theta Parameters to estimate        (n-by-1 matrix)
//! grad  Gradient of the cost function (n-by-1 matrix)

#![allow(non_snake_case)]
#![deny(warnings)]
#![feature(plugin)]
#![plugin(linalg_macros)]

extern crate cast;
extern crate env_logger;
extern crate linalg;
extern crate lines;
extern crate time;

#[macro_use]
extern crate log;

use std::fs::File;
use std::io::{BufReader, self};
use std::path::Path;

use cast::From as _0;
use linalg::prelude::*;
use linalg::{Col, ColMut, Transposed, SubMat};
use lines::Lines;

macro_rules! timeit {
    ($msg:expr, $e:expr) => {{
        let now = time::precise_time_ns();
        let out = $e;
        let elapsed = time::precise_time_ns() - now;
        println!(concat!($msg, " took {} ms"), f64::from_(elapsed) / 1_000_000.);
        out
    }}
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Label {
    Setosa,
    Versicolor,
    Virginica,
}

impl<'a> From<&'a str> for Label {
    fn from(string: &str) -> Label {
        use self::Label::*;

        match string {
            "Iris-setosa" => Setosa,
            "Iris-versicolor" => Versicolor,
            "Iris-virginica" => Virginica,
            _ => unreachable!(),
        }
    }
}

fn main() {
    env_logger::init().unwrap();

    // Some dummy operation to force the initialization of OpenBLAS' runtime (~90 ms) here rather
    // than during the measurements below
    (&mat![1., 2.; 3., 4.].inv() * &mat![1., 2.; 3., 4.]).eval();

    let (data, ref y) = timeit!("Loading data", {
        load("iris.csv").unwrap()
    });

    let (m, n) = data.size();

    let ref mut X = Mat::ones((m, n + 1));
    X[.., 1..] = data[.., ..n];

    let alpha = 0.01;
    let max_niters = 100_000;

    let ref mut setosa = ColVec::zeros(n + 1);
    let iters = timeit!("\nSetosa vs rest", {
        descent(Label::Setosa, X, y, setosa, alpha, max_niters)
    });
    println!("Estimated parameters: {:?}", setosa);
    println!("Iterations required: {}\n", iters);

    let ref mut versicolor = ColVec::zeros(n + 1);
    let iters = timeit!("Versicolor vs rest", {
        descent(Label::Versicolor, X, y, versicolor, alpha, max_niters)
    });
    println!("Estimated parameters: {:?}", versicolor);
    println!("Iterations required: {}\n", iters);

    let ref mut virginica = ColVec::zeros(n + 1);
    let iters = timeit!("Virginica vs rest", {
        descent(Label::Virginica, X, y, virginica, alpha, max_niters)
    });
    println!("Estimated parameters: {:?}", virginica);
    println!("Iterations required: {}\n", iters);
}

/// Performs the gradient descent algorithm to find the value of `theta` that minimizes the cost
/// function.
///
/// positive    Label    Positive class for one-vs-rest classifier
/// X           (m, n)
/// y           (m, 1)
/// theta       (n, 1)
/// alpha       scalar   Step size
/// max_niters  integer  Maximum number of iterations
///
/// -> Returns the number of iterations required to converge to a solution
fn descent(
    positive: Label,
    X: &SubMat<f64>,
    y: &Col<Label>,
    theta: &mut Col<f64>,
    alpha: f64,
    max_niters: u32,
) -> u32 {
    const TOL: f64 = 1e-5;

    // Pre-allocate column vectors to avoid allocations in the loop
    let ref mut z = ColVec::zeros(X.nrows());
    let ref mut grad = ColVec::zeros(theta.nrows());

    let mut last_J = cost(positive, theta, X, y, z, grad);
    for i in 0..max_niters {
        *theta -= alpha * grad;

        let J = cost(positive, theta, X, y, z, grad);

        debug!("i: {}, J: {}, theta: {:?}", i, J, theta);

        // Stop condition: `cost` reduced by less than `TOL`% in last iteration
        if (J - last_J).abs() / J.max(last_J) < TOL {
            return i
        }

        last_J = J;
    }

    max_niters
}


/// positive  Label   Positive class for one-vs-rest classifier
/// X         (m, n)
/// theta     (n, 1)
/// y         (m, 1)
/// z         (m, 1)  Auxiliary buffer
/// grad      (n, 1)  Gradient of the cost function
fn cost(
    positive: Label,
    theta: &Col<f64>,
    X: &SubMat<f64>,
    y: &Col<Label>,
    z: &mut Col<f64>,
    grad: &mut Col<f64>,
) -> f64 {
    fn sigmoid(z: &mut Col<f64>) {
        for x in z {
            *x = 1. / (1. + (-*x).exp());
        }
    }

    let m = f64::from_(y.nrows());

    // z = h = sigmoid(X * theta)
    z[..] = X * theta;
    sigmoid(z);

    let J = y.iter().zip(&*z).map(|(&y, h)| {
        if y == positive {
            -h.ln()
        } else {
            -(1. - h).ln()
        }
    }).fold(0., |x, y| x + y) / m;

    // z = h - y
    for (&y, z) in y.iter().zip(z) {
        if y == positive {
            *z -= 1.
        }
    }

    // gradient = X' * (h - y) / m
    grad[..] = X.t() * z / m;

    J
}

/// Loads data from a CSV file
fn load<P>(path: P) -> io::Result<(Transposed<Mat<f64>>, ColVec<Label>)> where P: AsRef<Path> {
    fn load(path: &Path) -> io::Result<(Transposed<Mat<f64>>, ColVec<Label>)> {
        let mut lines = Lines::from(BufReader::new(try!(File::open(path))));

        let mut labels = vec![];
        let mut v = vec![];

        let ncols = {
            let mut ncols = 0;

            for item in try!(lines.next().unwrap()).split(',') {
                match item.parse() {
                    Err(_) => labels.push(Label::from(item)),
                    Ok(number) => {
                        ncols += 1;
                        v.push(number);
                    },
                }
            }

            ncols
        };

        let mut nrows = 1;
        while let Some(line) = lines.next() {
            let line = try!(line);

            if line.is_empty() {
                continue
            }

            for item in line.split(',') {
                match item.parse() {
                    Err(_) => labels.push(Label::from(item)),
                    Ok(number) => v.push(number),
                }
            }

            nrows += 1;
        }

        unsafe {
            Ok((
                Mat::from_raw_parts(v.into_boxed_slice(), (ncols, nrows)).t(),
                ColVec::new(labels.into_boxed_slice()),
            ))
        }
    }

    load(path.as_ref())
}
