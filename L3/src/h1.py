import numpy as np

X = [[0.2, 0.7], [0.3, 0.3], [0.4, 0.5], [0.6, 0.5], [0.1, 0.4], [0.4, 0.6], [0.6, 0.2], [0.7, 0.4], [0.8, 0.6],
     [0.7, 0.5]]
Y = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]

X = np.array(X)
Y = np.array(Y)

X_p = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose())
print(np.dot(X_p, Y))
