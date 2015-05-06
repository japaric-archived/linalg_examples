import numpy
import time


def cost(X, y, theta):
    m = len(y)
    e = X.dot(theta) - y
    return e.T.dot(e)[0, 0] / 2 / m


def normalize(X):
    mu = numpy.mean(X, axis=0)
    sigma = numpy.std(X, axis=0)
    X -= mu[None, :]
    X /= sigma[None, :]

    return (mu, sigma)


def descent(X, y, theta, alpha, max_niters):
    TOL = 1e-5

    m = len(y)

    last_J = cost(X, y, theta)
    for i in range(0, max_niters):
        e = X.dot(theta) - y
        theta -= alpha / m * X.T.dot(e)

        J = cost(X, y, theta)

        if abs(J - last_J) / max(J, last_J) < TOL:
            return i

        last_J = J

    return max_niters

# Dummy operation to force initialization of OpenBLAS' runtime
numpy.linalg.inv(numpy.array([[1, 2], [3, 4]])).dot(
    numpy.array([[1, 2], [3, 4]]))

now = time.time()
data = numpy.loadtxt(open('mpg.tsv', 'rb'))
elapsed = time.time() - now
print("Loading data took {} ms".format(elapsed * 1000))

m = data.shape[0]
print("{} observations".format(m))

n = data.shape[1] - 1
print("{} independent variables\n".format(n))

y = data[:, 0]
X = numpy.c_[numpy.ones((m, 1)), data[:, 1:]]
y.shape = (m, 1)  # make sure this an array and not a vector

now = time.time()
(mu, sigma) = normalize(X[:, 1:])
elapsed = time.time() - now
print("Normalization took {} ms".format(elapsed * 1000))
print("mean: {}".format(mu))
print("std deviation: {}\n".format(sigma))

theta = numpy.zeros((n + 1, 1))
alpha = 0.01
max_niters = 100000

now = time.time()
niters = descent(X, y, theta, alpha, max_niters)
elapsed = time.time() - now
print("Gradient descent took {} ms".format(elapsed * 1000))

print("Estimated parameters: {}".format(theta.ravel()))
print("Iterations required: {}".format(niters))
