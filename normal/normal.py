import numpy
import time

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

theta = numpy.zeros((n + 1, 1))
alpha = 0.01

now = time.time()
theta = numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
elapsed = time.time() - now
print("Solving the normal equation took {} ms".format(elapsed * 1000))

print("Estimated parameters: {}".format(theta.ravel()))
