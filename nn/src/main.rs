#![deny(warnings)]

// User configurable parameters

// The training set contains 60.000 examples, you can work on the subset `OFFSET..(OFFSET+SIZE)`
const TRAINING_SET_OFFSET: u32 = 0;
const TRAINING_SET_SIZE: u32 = 1_000;

// The test set contains 10.000 examples, you can work on the subset `OFFSET..(OFFSET+SIZE)`
const TEST_SET_OFFSET: u32 = 0;
const TEST_SET_SIZE: u32 = 10_000;

// Number of units in the hidden layer
const NUM_HIDDEN_UNITS: u32 = 30;

// For how long to train the neural network. 1 epoch = 1 pass through the whole training set
const EPOCHS: u32 = 1_000;

// The normalization parameter. Used to avoid overfitting. Setting it to zero disables
// normalization. Using a large normalization parameter may lead to underfitting
const LAMBDA: f64 = 0.1;

// How fast should the neural network learn. The bigger the learning rate, the faster the NN will
// "descent" to a minima. A value too large may cause the NN to "step over" a minima.
const INITIAL_LEARNING_RATE: f64 = 0.3;

// Momentum can help to converge faster, and reduces side to side "oscillations". Typical values
// are in the range of [0, 1]. A value of 0 disables momentum completely.
const MOMENTUM: f64 = 0.9;

// Don't touch anything below this point
extern crate byteorder;
extern crate cast;
extern crate image;
extern crate linalg;
extern crate rand;
extern crate time;

mod images;
mod labels;
mod network;

use std::cmp;

use cast::From as _0;
use linalg::Buffer;
use linalg::prelude::*;

use images::Images;
use labels::Labels;
use network::{Network, Options};

const TRAINING_SET_IMAGES: &'static str = "train-images-idx3-ubyte";
const TRAINING_SET_LABELS: &'static str = "train-labels-idx1-ubyte";

const TEST_SET_IMAGES: &'static str = "t10k-images-idx3-ubyte";
const TEST_SET_LABELS: &'static str = "t10k-labels-idx1-ubyte";

macro_rules! timeit {
    ($msg:expr, $e:expr) => {{
        let now = time::precise_time_ns();
        let out = $e;
        let elapsed = time::precise_time_ns() - now;
        println!(concat!($msg, " took {} s"), f64::from_(elapsed) / 1_000_000_000.);
        out
    }}
}

fn main() {
    assert!(TRAINING_SET_SIZE >= 100);
    assert!(TEST_SET_SIZE >= 100);

    println!("Number of hidden units: {}", NUM_HIDDEN_UNITS);
    println!("Normalization parameter: {}", LAMBDA);
    println!("Initial learning rate: {}", INITIAL_LEARNING_RATE);
    println!("Momentum: {}\n", MOMENTUM);

    let training_subset = TRAINING_SET_OFFSET..TRAINING_SET_SIZE;

    // Load training set
    let training_images = Images::load(TRAINING_SET_IMAGES, training_subset.clone()).unwrap();;
    let training_labels = Labels::load(TRAINING_SET_LABELS, training_subset.clone()).unwrap();;

    // Load test set
    let test_subset = TEST_SET_OFFSET..TEST_SET_SIZE;
    let test_images = Images::load(TEST_SET_IMAGES, test_subset.clone()).unwrap();;
    let ref test_labels = Labels::load(TEST_SET_LABELS, test_subset.clone()).unwrap();;

    // "Show" a random sample of the training set
    println!("Storing a sample of the training set to training_set.png\n");
    let ref mut rng = rand::thread_rng();
    let indices = rand::sample(rng, training_subset, 100);
    training_images.save(indices.iter().map(|&x| x), "training_set.png").unwrap();

    // Allocate a chunk of memory now to avoid calling the allocator during training/validation
    let m = training_images.size();
    let i = training_images.num_pixels();
    let h = NUM_HIDDEN_UNITS;
    let o = training_labels.num_classes();
    let ref mut buffer = Buffer::new({
        let m = usize::from_(m);
        let h = usize::from_(h);
        let o = usize::from_(o);

        let training = m * (2 * h + 2 * o + h + 1) + 3 * o;

        let m = usize::from_(test_images.size());
        let test = m * (h + o + 1);

        cmp::max(training, test)
    });

    let inputs = training_images.to_dataset();
    let targets = training_labels.to_dataset();

    // A new untrained network
    let mut nn = Network::new(i, h, o);

    println!("The untrained NN classified the first row of the sample as:\n{:?}\n",
        indices.iter().take(10).map(|&i| {
            nn.classify(inputs.row(i), buffer)
        }).collect::<Vec<_>>());

    println!("Training the NN with {} examples", TRAINING_SET_SIZE);

    timeit!("Training",
        nn.train(inputs.slice(..), targets.slice(..), buffer, Options {
            epochs: EPOCHS,
            lambda: LAMBDA,
            learning_rate: INITIAL_LEARNING_RATE,
            momentum: MOMENTUM,
        }));

    println!("\nThe trained NN now classifies the first row of the sample as:\n{:?}\n",
        indices.iter().take(10).map(|&i| {
            nn.classify(inputs.row(i), buffer)
        }).collect::<Vec<_>>());

    let test_inputs = test_images.to_dataset();

    println!("Validating NN with {} examples", TEST_SET_SIZE);
    let errors = timeit!("Validation", {
        nn.validate(test_inputs.slice(..), test_labels, buffer)
    });

    let rate = 100. * f64::from_(errors.len()) / f64::from_(TEST_SET_SIZE);

    println!("\n{} of {} examples were misclassified ({:.2}% error rate)\n",
             errors.len(), TEST_SET_SIZE, rate);

    println!("Storing a sample of the misclassified digits to errors.png\n");
    test_images.save(errors.iter().map(|&x| x), "errors.png").unwrap();

    println!("The first row of the sample was misclassified as:\n{:?}\n",
        errors.iter().take(10).map(|&i| {
            nn.classify(test_inputs.row(i), buffer)
        }).collect::<Vec<_>>());

    println!("Correct labels were:\n{:?}",
        (0..10).map(|i| test_labels[i]).collect::<Vec<_>>());
}
