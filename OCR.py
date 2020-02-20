#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

(x_train, y_train), (x_test, y_test)  = mnist.load_data()
print (x_train.shape)

aux = np.random.randint(0,len(x_train))
plt.imshow(x_train[aux], cmap=plt.get_cmap('gray'))
plt.show()

### Normalization
rows = x_train[0].shape[0]
cols = x_train[0].shape[1]

x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

den = 255

x_train /= den
x_test /= den

# ### Encoding

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

classes = y_test.shape[1]

## Training the Model

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(rows, cols, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = SGD(0.01), metrics = ['accuracy'])

print(model.summary())

epochs = 1

fitted = model.fit(x_train, y_train, batch_size = 32, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))

result = model.evaluate(x_test, y_test, verbose=0)
print('[Test Loss]: ', result[0])
print('[Test Accuracy]: ', result[1])


### Step 6 - Ploting our Loss and Accuracy Charts



fitted_dict = fitted.history

losses = fitted_dict['loss']
val_losses = fitted_dict['val_loss']
epochs = range(1, len(losses) + 1)

plot1 = plt.plot(epochs, val_losses, label='Test Loss')
plot2 = plt.plot(epochs, losses, label='Train Loss')
plt.setp(plot1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(plot2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

fitted_dict = fitted.history

acc = fitted_dict['acc']
val_acc = fitted_dict['val_acc']
epochs = range(1, len(losses) + 1)

plot1 = plt.plot(epochs, acc, label='Test Accuracy')
plot2 = plt.plot(epochs, val_acc, label='Train Accuracy')
plt.setp(plot1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(plot2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

### Saving the Model and architecture image

model.save("models/ocr_cnn.h5")


### Load the Model and testing it

clf = load_model('models/ocr_cnn.h5')

for i in range(0,10):
    aux = np.random.randint(0,len(x_test))
    image = x_test[aux]

    image_N = cv2.resize(image, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
    image = image.reshape(1,28,28,1)

    prediction = str(clf.predict_classes(image, 1, verbose = 0)[0])

    # drawing
    expanded = cv2.copyMakeBorder(image_N, 0, 0, 0, image_N.shape[0] ,cv2.BORDER_CONSTANT,value=[0,0,0])
    expanded = cv2.cvtColor(expanded, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded, str(prediction), (152, 70) , cv2.FONT_HERSHEY_COMPLEX_SMALL,4, (0,255,0), 2)
    cv2.imshow("Prediction", expanded)

    cv2.waitKey(0)

cv2.destroyAllWindows()
