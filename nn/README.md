# Neural Network

Training and validating a neural network (NN) for identification of hand written digits.

This is a classification problem, the input is a 28x28 grayscale image, and the output is a label
that ranges from 0 to 9.

The NN is a MLP (multi layer perceptron) with a single hidden layer, trained using batch processing
and gradient descent with adaptive learning rate.

The database used for training and validation can be found [here]. It consists of a training set of
60,000 examples and a test set of 10,000 examples.

[here]: http://yann.lecun.com/exdb/mnist/

To run this example, use the following commands:

(Be sure to have libblas and liblapack installed)

```
# WARNING! Downloads ~10 MB of compressed data, which then gets uncompressed to ~50 MB
$ ./fetch_data.sh
$ cargo run --release
```

Feel free to experiment by changing the "constants" at the top of the `src/main.rs` file.

Here's an example output that uses the whole database.

```
Number of hidden units: 300
Normalization parameter: 0.1
Initial learning rate: 0.3
Momentum: 0.9

Storing a sample of the training set to training_set.png
```

![training set](/training_set.png)

```
The untrained NN classified the first row of the sample as:
[8, 5, 5, 3, 8, 5, 3, 7, 3, 7]

Training the NN with 60000 examples
Epochs MSE    LR
0      9.7303 0.3000
2      5.3440 0.0827
(..)
997    0.0428 5.4309
999    0.0426 5.9876
Training took 3797.647199067 s

The trained NN now classifies the first row of the sample as:
[0, 8, 2, 7, 0, 8, 4, 0, 9, 2]

Validating NN with 10000 examples
Validation took 0.15985069 s

186 of 10000 examples were misclassified (1.86% error rate)

Storing a sample of the misclassified digits to errors.png
```

![errors](/errors.png)

```
The first row of the sample was misclassified as:
[9, 9, 6, 7, 3, 7, 0, 5, 2, 7]

Correct labels were:
[4, 2, 4, 2, 5, 3, 6, 3, 8, 2]
```
