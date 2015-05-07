import csv
import numpy
import time

MAP = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}


def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))


def cost(label, theta, X, y):
    tag = MAP[label]

    y = numpy.array([1. if label == tag else 0. for label in y])
    m = len(y)
    y.shape = (m, 1)

    h = sigmoid(X.dot(theta))

    J = (-y.T.dot(numpy.log(h)) - (1 - y).T.dot(numpy.log(1 - h)))[0, 0] / m

    grad = X.T.dot(h - y) / m

    return J, grad


def descent(label, X, y, theta, alpha, max_niters):
    TOL = 1e-5

    last_J, grad = cost(label, theta, X, y)
    for i in range(0, max_niters):
        theta -= alpha * grad

        J, grad = cost(label, theta, X, y)

        if abs(J - last_J) / max(J, last_J) < TOL:
            return i

        last_J = J

    return max_niters

# Dummy operation to force initialization of OpenBLAS' runtime
numpy.linalg.inv(numpy.array([[1, 2], [3, 4]])).dot(
    numpy.array([[1, 2], [3, 4]]))

now = time.time()
X = []
y = []

with open('iris.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=',')

    for row in reader:
        if row:
            X.append(map(float, row[:-1]))
            y.append(MAP[row[-1]])

X = numpy.array(X)

y = numpy.array(y)
y.shape = (len(y), 1)

elapsed = time.time() - now
print("Loading data took {} ms\n".format(elapsed * 1000))

n = X.shape[1]

X = numpy.c_[numpy.ones((X.shape[0], 1)), X]

alpha = 0.01
max_niters = 100000

setosa = numpy.zeros((n + 1, 1))
now = time.time()
iters = descent("Iris-setosa", X, y, setosa, alpha, max_niters)
elapsed = time.time() - now
print("Setosa vs rest took {} ms".format(elapsed * 1000))

print("Estimated parameters: {}".format(setosa.ravel()))
print("Iterations required: {}\n".format(iters))

versicolor = numpy.zeros((n + 1, 1))
now = time.time()
iters = descent("Iris-versicolor", X, y, versicolor, alpha, max_niters)
elapsed = time.time() - now
print("Versicolor vs rest took {} ms".format(elapsed * 1000))

print("Estimated parameters: {}".format(versicolor.ravel()))
print("Iterations required: {}\n".format(iters))

virginica = numpy.zeros((n + 1, 1))
now = time.time()
iters = descent("Iris-virginica", X, y, virginica, alpha, max_niters)
elapsed = time.time() - now
print("Virginica vs rest took {} ms".format(elapsed * 1000))

print("Estimated parameters: {}".format(virginica.ravel()))
print("Iterations required: {}".format(iters))
