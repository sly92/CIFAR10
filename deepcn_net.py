#!/usr/bin/env python3
# -*- coding: utf-8 -*-


""" Convnet for the CIFAR 10 Project """


# from pprint import pprint
# from random import randint
from numpy import reshape
from keras import Model
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import sgd
from keras.callbacks import TensorBoard
from keras.activations import relu, sigmoid
from keras.metrics import categorical_accuracy, mse
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten


experiment_name = 'CIPHAR10_DEEPCONVNET_3_128CONV_MAXPOOL_22_256DENSE_LR_05'
tb_callback = TensorBoard('./logs/' + experiment_name)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Transformation to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2D Convolution
x_train = reshape(x_train, (-1, 32, 32, 3)) / 255.0
x_test = reshape(x_test, (-1, 32, 32, 3)) / 255.0

inputs = Input((32, 32, 3))
hidden_conv = Conv2D(128, (3, 3), activation=relu, padding='same')(inputs)
pool_layers = MaxPooling2D(pool_size=(2, 2))(hidden_conv)

for i in range(1, 3, 1):
    hidden_conv = Conv2D(128, (3, 3), activation=relu, padding='same')(pool_layers)
    pool_layers = MaxPooling2D(pool_size=(2, 2))(hidden_conv)

flatten = Flatten()(pool_layers)
hidden_dense = Dense(256, activation=relu)(flatten)
outputs = Dense(10, activation=sigmoid)(hidden_dense)

model = Model(inputs, outputs)
model.compile(sgd(lr=0.5), mse, metrics=[categorical_accuracy])
model.fit(x_train, y_train,
          batch_size=256,
          epochs=120,
          callbacks=[tb_callback],
          validation_data=(x_test, y_test))

model.save('./model_save/deep_convnet/' + experiment_name + '.h5')
plot_model(model, to_file='./model_plot/deep_convnet/' + experiment_name + '.png',
           show_shapes=True, show_layer_names=True)
