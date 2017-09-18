import numpy as np
X_Test = np.loadtxt(open("foo.csv", "rb"), delimiter = ",", skiprows=0)
Y_Test = np.argmax(X_Test, axis = 1)
np.savetxt("foo1.csv", Y_Test, fmt='%d', delimiter=",")