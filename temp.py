import numpy as np, numpy.random

np.set_printoptions(precision=2)
pre_trained_data = np.random.dirichlet(np.ones(3), size=10)
pre_trained_data = [np.append(data, i%3) for i, data in enumerate(pre_trained_data)]
print(pre_trained_data)