
from __future__ import print_function
import keras
from keras.datasets import cifar10
from utils import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
model = load_model('my_model1.h5')
X_Test = np.loadtxt(open("dataset/flipped_test.csv", "rb"), delimiter = ",", skiprows=0)
nn = X_Test.shape[0]
X_Test = np.reshape(X_Test, [nn, 32,32,3])
Y_Test = model.predict(X_Test)
np.savetxt("foo.csv", Y_Test, fmt='%d', delimiter=",")