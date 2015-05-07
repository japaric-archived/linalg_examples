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
//! Solver: Normal Equation
//!
//! ```
//! theta = (X' * X)^-1 * X' * y
//!
//! E                (m-by-1 matrix)
//! X                (m-by-n matrix)
//! Y                (m-by-1 matrix)
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
extern crate time;

#[macro_use]
extern crate log;

use std::fs::File;
use std::io::{BufReader, self};
use std::path::Path;

use cast::From as _0;
use linalg::prelude::*;
use linalg::Transposed;
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

    let ref mut X = Mat::ones((m, n + 1));
    X[.., 1..] = data[.., 1..];
    let y = data.col(0);

    let X = &*X;

    let theta = timeit!("Solving the normal equation", {
        (&(X.t() * X).inv() * X.t() * y).eval()
    });

    println!("Estimated parameters: {:?}", theta);
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
