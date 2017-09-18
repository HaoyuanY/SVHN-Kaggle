from __future__ import print_function
import keras
from keras.datasets import cifar10
from utils import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

ModelName = "BatchNorm_Layers_LRELU_ADAM_3Affine"
batch_size = 100
num_classes = 10
epochs = 30
data_augmentation = False
l_rate = 0.1
l_decay = 1e-6
fn_1 = 32

output = False
# The data, shuffled and split between train and test sets:
"""
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
"""
x_train, y_train = load_my_train_data(data_type = "test_long")
y_train[y_train == 10] = 0
print(x_train.shape)

n = x_train.shape[0]
x_train = np.reshape(x_train, [n, 32,32,3])
print(x_train.shape)
print("Training data loaded!")
if output:
    val_size = 0.1
else:
    val_size = 0.2
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = val_size, random_state=1)
# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (5, 5), padding='VALID',
                 input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (1, 1)))
model.add(Conv2D(32, (1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='VALID'))
model.add(Conv2D(64, (1, 1)))
model.add(Conv2D(64, (1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (2, 2), padding='VALID'))
model.add(Conv2D(128, (1, 1)))
model.add(Conv2D(128, (1, 1)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.Adam(lr=l_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=l_decay)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
if not data_augmentation:
  print("there")
  print('Not using data augmentation.')
  model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
  model.predict_classes(x_test, verbose = 1)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))
    save_dir = create_output_dir(ModelName)
    os.chdir(save_dir)

    model.save(ModelName+'.h5')

    if output:
        X_Test = np.loadtxt(open("../Raw_Data/flipped_data/flipped_test.csv", "rb"), delimiter = ",", skiprows=0)
        n = X_Test.shape[0]
        X_Test = np.reshape(X_Test, [n, 32,32,3])
        Y_Test = model.predict(X_Test)
        format_output(Y_test, ModelName)
    else:
        # history = model.fit_generator(datagen.flow(x_train, y_train,
        #                                  batch_size=batch_size),
        #                     steps_per_epoch=x_train.shape[0] // batch_size,
        #                     epochs=epochs,
        #                     validation_data=(x_test, y_test))

        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['test', 'train'], loc='upper left')
        plt.savefig(ModelName+"_acc.jpg")
        # summarize history for loss
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['test', 'train'], loc='upper left')
        plt.savefig(ModelName+"_loss.jpg")