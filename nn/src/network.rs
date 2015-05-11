use cast::From as _0;
use linalg::prelude::*;
use linalg::{Buffer, Row, SubMat, SubMatMut};
use rand::{Rng, XorShiftRng, self};
use time;

use labels::Labels;

pub struct Options {
    /// Normalization parameter (weight decay)
    pub epochs: u32,
    pub lambda: f64,
    pub learning_rate: f64,
    pub momentum: f64,
}

/// A neural network with a single hidden layer
pub struct Network {
    /// Number of units per layer
    pub s: (u32, u32, u32),
    /// Parameters
    pub theta: Box<[f64]>,
}

impl Network {
    /// New untrained network
    pub fn new(num_inputs: u32, num_hidden_units: u32, num_outputs: u32) -> Network {
        assert!(num_outputs > 0);

        let i = num_inputs;
        let h = num_hidden_units;
        let o = num_outputs;

        let n = usize::from_(i + 1) * usize::from_(h) + usize::from_(h + 1) * usize::from_(o);

        let ref mut rng: XorShiftRng = rand::thread_rng().gen();

        let theta: Vec<_> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();

        Network {
            s: (i, h, o),
            theta: theta.into_boxed_slice(),
        }
    }

    /// Classifies an image
    pub fn classify(&self, image: Row<f64>, buffer: &mut Buffer<f64>) -> u32 {
        let (i, h, o) = self.s;

        assert_eq!(image.ncols(), i + 1);

        let mut pool = buffer.as_pool();

        let (theta_1, theta_2) = split_theta(&self.theta, self.s);
        let mut a_2 = pool.row((h + 1));
        let mut h = pool.row((o));

        // Feed forward
        let a_1 = image;

        // a_2 = [ones(m, 1), g(a_1 * theta_1')]
        a_2[0] = 1.;
        a_2.slice_mut(1..).set(a_1 * theta_1.t());
        g(a_2.slice_mut((1..)));

        // h = g(a_2 * theta_2)
        h.set(a_2.slice(..) * theta_2.t());
        g(h.slice_mut(..));

        let mut iter = h.iter().zip(0..);
        let (mut max, mut i) = iter.next().unwrap();

        for (x, j) in iter {
            if x > max {
                max = x;
                i = j;
            }
        }

        i
    }

    /// Trains the neural network
    pub fn train(
        &mut self,
        images: SubMat<f64>,
        labels: SubMat<f64>,
        buffer: &mut Buffer<f64>,
        options: Options)
    {
        const TOL: f64 = 1e-5;
        const TICK: u64 = 5_000_000_000;

        assert!(images.col(0).iter().all(|x| x.eq(&1.)));

        let mut alpha = options.learning_rate;
        let epochs = options.epochs;
        let lambda = options.lambda;
        let m = options.momentum;
        let s = self.s;
        let x = images;
        let y = labels;

        let (mut theta_1, mut theta_2) = split_theta_mut(&mut self.theta, s);

        let mut prev_step_1 = Mat::zeros(theta_1.size());
        let mut prev_step_2 = Mat::zeros(theta_2.size());
        let mut prev_theta_1 = Mat::zeros(theta_1.size());
        let mut prev_theta_2 = Mat::zeros(theta_2.size());
        let mut grad_1 = Mat::zeros(theta_1.size());
        let mut grad_2 = Mat::zeros(theta_2.size());

        let mut last = time::precise_time_ns().wrapping_sub(TICK);
        println!("Epochs MSE    LR");
        for i in 0..epochs+1 {
            let cost =  cost_plus_grad(
                theta_1.slice(..), theta_2.slice(..), x, y, s, lambda, grad_1.slice_mut(..),
                grad_2.slice_mut(..), buffer);

            let now = time::precise_time_ns();

            if now.wrapping_sub(last) > TICK {
                last = now;
                println!("{:<6} {:<6.4} {:.4}", i, cost, alpha);
            }

            prev_theta_1.set(theta_1.slice(..));
            prev_theta_2.set(theta_2.slice(..));

            loop {
                theta_1.sub_assign(alpha * grad_1.slice(..));
                theta_1.add_assign(m * prev_step_1.slice(..));

                theta_2.sub_assign(alpha * grad_2.slice(..));
                theta_2.add_assign(m * prev_step_2.slice(..));

                let new_cost =
                    cost_(theta_1.slice(..), theta_2.slice(..), x, y, s, lambda, buffer);

                if (new_cost - cost).abs() / new_cost.max(cost) < TOL {
                    return println!("{:<6} {:<6.4} {:.4} (local minima)", i, cost, alpha);
                } else if new_cost < cost {
                    // Accelerate
                    alpha += 0.05 * alpha;

                    break
                } else {
                    // Rollback
                    alpha /= 2.;

                    theta_1.set(prev_theta_1.slice(..));
                    theta_2.set(prev_theta_2.slice(..));

                    prev_step_1.set(0.);
                    prev_step_2.set(0.);
                }
            }

            prev_step_1.set(theta_1.slice(..));
            prev_step_1.sub_assign(prev_theta_1.slice(..));

            prev_step_2.set(theta_2.slice(..));
            prev_step_2.sub_assign(prev_theta_2.slice(..));
        }
    }

    pub fn validate(
        &self,
        images: SubMat<f64>,
        labels: &Labels,
        buffer: &mut Buffer<f64>,
    ) -> Vec<u32> {
        let (_, h, o) = self.s;
        let m = images.nrows();
        let mut pool = buffer.as_pool();

        let (theta_1, theta_2) = split_theta(&self.theta, self.s);
        let mut a_2 = pool.mat((m, (h + 1)));
        let mut h = pool.mat((m, o));

        // Feed forward
        let a_1 = images;

        // a_2 = [ones(m, 1), g(a_1 * theta_1')]
        a_2.col_mut(0).set(1.);
        a_2.slice_mut((.., 1..)).set(a_1 * theta_1.t());
        g(a_2.slice_mut(((.., 1..))));

        // h = g(a_2 * theta_2)
        h.set(a_2.slice(..) * theta_2.t());
        g(h.slice_mut(..));

        h.rows().zip(0..).filter_map(|(h, i)| {
            let max = h[u32::from_(labels[i])];

            if h.iter().all(|&x| x <= max) {
                None
            } else {
                Some(i)
            }

        }).collect()
    }
}

/// Evaluates cost function
pub fn cost_<'a>(
    theta_1: SubMat<f64>,  // Input. (h, i + 1)
    theta_2: SubMat<f64>,  // Input. (o, h + 1)
    x: SubMat<f64>,      // Input: (m, i + 1)
    y: SubMat<f64>,      // Input: (m, o)
    s: (u32, u32, u32),
    lambda: f64,
    buffer: &'a mut Buffer<f64>,
) -> f64 {
    let (_, h, o) = s;
    let m = y.nrows();

    let mut pool = buffer.as_pool();

    let mut a_2 = pool.mat((m, h + 1));
    let mut a_3 = pool.mat((m, o));
    let mut z_2 = pool.mat((m, h));

    // Feed forward
    debug_assert!(x.col(0).iter().all(|x| x.eq(&1.)));

    let a_1 = x;
    let m_ = f64::from_(m);

    // z_2 = a_1 * theta_1'
    z_2.set(a_1 * theta_1.t());

    // a_2 = [ones(m, 1), g(z_2)]
    a_2.col_mut(0).set(1.);
    a_2.slice_mut((.., 1..)).set(z_2.slice(..));
    g(a_2.slice_mut((.., 1..)));

    // h = a_3 = g(a_2 * theta_2)
    a_3.set(a_2.slice(..) * theta_2.t());
    g(a_3.slice_mut(..));

    // unnormalized_cost = sum(-y .* log(h) - (1 - y) .* log(1 - h)) / m
    // (BLAS doesn't provide element-wise product or sum all elements routines, so I'll use a for
    // loop here)
    let mut cost = 0.;

    let mut _1_y = pool.row(o);
    let mut log_h = pool.col(o);
    let mut log_1_h = pool.col(o);

    // TODO this loop can be fork-join parallelized
    for (h, y) in a_3.rows().zip(y.rows()) {
        let h = h.t();

        // 1 - y
        _1_y.set(1.);
        _1_y.sub_assign(y);

        // log(h)
        log_h.set(h);
        log(log_h.slice_mut(..));

        // log(1 - h)
        log_1_h.set(1.);
        log_1_h.sub_assign(h);
        log(log_1_h.slice_mut(..));

        cost -= y * &log_h + &_1_y * &log_1_h;
    }

    // Normalization
    // cost = unnormalized_cost + lambda * (ssq(theta_1[:, 1:]) + ssq(theta_2[:, 1:])) / 2 / m
    let ssq = |m: SubMat<f64>| {
        let norm = m.slice((.., 1..)).norm();
        norm * norm
    };

    let ssq = ssq(theta_1) + ssq(theta_2);

    cost += lambda * ssq / 2.;
    cost /= m_;

    cost
}

/// Evaluates cost function and its gradients
pub fn cost_plus_grad(
    theta_1: SubMat<f64>,  // Input. (h, i + 1)
    theta_2: SubMat<f64>,  // Input. (o, h + 1)
    x: SubMat<f64>,      // Input: (m, i + 1)
    y: SubMat<f64>,      // Input: (m, o)
    s: (u32, u32, u32),
    lambda: f64,
    mut grad_1: SubMatMut<f64>,  // Output. (h, i + 1)
    mut grad_2: SubMatMut<f64>,  // Output. (o, h + 1)
    buffer: &mut Buffer<f64>,
) -> f64 {
    let (_, h, o) = s;
    let m = y.nrows();

    let mut pool = buffer.as_pool();

    let mut a_2 = pool.mat((m, h + 1));
    let mut a_3 = pool.mat((m, o));
    let mut z_2 = pool.mat((m, h));

    // Feed forward
    debug_assert!(x.col(0).iter().all(|x| x.eq(&1.)));

    let a_1 = x;
    let m_ = f64::from_(m);

    // z_2 = a_1 * theta_1'
    z_2.set(a_1 * theta_1.t());

    // a_2 = [ones(m, 1), g(z_2)]
    a_2.col_mut(0).set(1.);
    a_2.slice_mut((.., 1..)).set(z_2.slice(..));
    g(a_2.slice_mut((.., 1..)));

    // h = a_3 = g(a_2 * theta_2)
    a_3.set(a_2.slice(..) * theta_2.t());
    g(a_3.slice_mut(..));

    // unnormalized_cost = sum(-y .* log(h) - (1 - y) .* log(1 - h)) / m
    // (BLAS doesn't provide element-wise product or sum all elements routines, so I'll use a for
    // loop here)
    let mut cost = 0.;

    let mut _1_y = pool.row(o);
    let mut log_h = pool.col(o);
    let mut log_1_h = pool.col(o);

    // TODO this loop can be fork-join parallelized
    for (h, y) in a_3.rows().zip(y.rows()) {
        let h = h.t();

        // 1 - y
        _1_y.set(1.);
        _1_y.sub_assign(y);

        // log(h)
        log_h.set(h);
        log(log_h.slice_mut(..));

        // log(1 - h)
        log_1_h.set(1.);
        log_1_h.sub_assign(h);
        log(log_1_h.slice_mut(..));

        cost -= y * &log_h + &_1_y * &log_1_h;
    }

    // Normalization
    // cost = unnormalized_cost + lambda * (ssq(theta_1[:, 1:]) + ssq(theta_2[:, 1:])) / 2 / m
    let ssq = |m: SubMat<f64>| {
        let norm = m.slice((.., 1..)).norm();
        norm * norm
    };

    let ssq = ssq(theta_1) + ssq(theta_2);

    cost += lambda * ssq / 2.;
    cost /= m_;

    // Back propagation
    let mut delta_2 = pool.mat((m, h));
    let mut delta_3 = pool.mat((m, o));

    // delta_3 = a_3 - y
    delta_3.set(a_3.slice(..));
    delta_3.sub_assign(y);

    // D_2 = (delta_3.t() * a_2) / m
    grad_2.set(delta_3.slice(..).t() * &a_2);

    // delta_2 = delta_3 * theta_2[:, 1:] .* g'(z_2)
    let mut dgdz_z_2 = z_2;
    dgdz(dgdz_z_2.slice_mut(..));
    delta_2.set(&delta_3 * theta_2.slice((.., 1..)));
    delta_2.mul_assign(dgdz_z_2.slice(..));

    // D_1 = (delta_2.t() * a_1) / m
    grad_1.set(delta_2.slice(..).t() * a_1);

    // Normalization
    // D_i[:, 1:] += lambda * theta_1[:, 1:] / m
    grad_1.slice_mut((.., 1..)).add_assign(lambda * theta_1.slice((.., 1..)));
    grad_1.div_assign(m_);

    grad_2.slice_mut((.., 1..)).add_assign(lambda * theta_2.slice((.., 1..)));
    grad_2.div_assign(m_);

    cost
}

/// Natural logarithm
fn log<'a, I>(z: I) where I: IntoIterator<Item=&'a mut f64> {
    for x in z {
        *x = (*x).ln()
    }
}

/// Sigmoid function
fn g<'a, I>(z: I) where I: IntoIterator<Item=&'a mut f64> {
    for x in z {
        *x = 1. / (1. + (-*x).exp())
    }
}

/// Gradient of the sigmoid function
fn dgdz<'a, I>(z: I) where I: IntoIterator<Item=&'a mut f64> {
    for x in z {
        let g = 1. / (1. + (-*x).exp());
        *x = g * (1. - g)
    }
}

fn split_theta(
    theta: &[f64],
    (i, h, o): (u32, u32, u32),
) -> (SubMat<f64>, SubMat<f64>) {
    let at = usize::from_(i + 1) * usize::from_(h);
    let (left, right) = theta.split_at(at);

    (SubMat::reshape(left, (h, i + 1)), SubMat::reshape(right, (o, h + 1)))
}

fn split_theta_mut(
    theta: &mut [f64],
    (i, h, o): (u32, u32, u32),
) -> (SubMatMut<f64>, SubMatMut<f64>) {
    let at = usize::from_(i + 1) * usize::from_(h);
    let (left, right) = theta.split_at_mut(at);

    (SubMatMut::reshape(left, (h, i + 1)), SubMatMut::reshape(right, (o, h + 1)))
}
