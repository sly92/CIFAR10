#!/usr/bin/env python3
# -*- coding: utf-8 -*-


""" Convnet for the CIFAR 10 Project """


# from pprint import pprint
from numpy import reshape #, dot
from keras import Model
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import sgd
from keras.callbacks import TensorBoard
from keras.activations import relu, sigmoid
from keras.metrics import categorical_accuracy, mse
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten


experiment_name = 'CIPHAR10_CONVNET_128_MAXPOOL_1_33_256D_LR_01_DCY_1E-4_MMT_0.9_N_RELU'
tb_callback = TensorBoard('./logs/' + experiment_name)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Transformation to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x_train = dot(x_train[...,:3], [1/3, 1/3, 1/3])
# x_test = dot(x_test[...,:3], [1/3, 1/3, 1/3])

# 2D Convolution
x_train = reshape(x_train, (-1, 32, 32, 3)) / 255.0
x_test = reshape(x_test, (-1, 32, 32, 3)) / 255.0

inputs = Input((32, 32, 3))
hidden = Conv2D(128, (4, 4), padding='same', activation=relu)(inputs)

# for i in range(4, 1, -1):
#     hidden = Conv2D(16*i, (3, 3), activation=relu, padding='same')(hidden)

pool_layers = MaxPooling2D((3, 3))(hidden)
flatten = Flatten()(pool_layers)
hidden_dense = Dense(256, activation=relu)(flatten)
outputs = Dense(10, activation=sigmoid)(hidden_dense)

model = Model(inputs, outputs)
model.compile(sgd(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True),
              mse, metrics=[categorical_accuracy])
model.fit(x_train, y_train,
          batch_size=512,
          epochs=120,
          callbacks=[tb_callback],
          validation_data=(x_test, y_test))

model.save('./model_save/convnet/' + experiment_name + '.h5')
plot_model(model, to_file='./model_plot/convnet/' + experiment_name + '.png',
           show_shapes=True, show_layer_names=True)
