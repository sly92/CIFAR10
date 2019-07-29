#!/usr/bin/env python3
# -*- coding: utf-8 -*-


""" Linear Model for the CIFAR 10 Project """


# from pprint import pprint
# from random import randint
from numpy import reshape #, dot
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import sgd
from keras.callbacks import TensorBoard
from keras.activations import sigmoid
from keras.layers import Dense
from keras.metrics import categorical_accuracy, mse


experiment_name = 'CIPHAR10_LINEAR_1024_TO_10_LR_05'
tb_callback = TensorBoard('./logs/' + experiment_name)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
cifar10_img_size = 32*32

# Transformation to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x_train = dot(x_train[...,:3], [1/3, 1/3, 1/3])
# x_test = dot(x_test[...,:3], [1/3, 1/3, 1/3])

# 2D Convolution
x_train = reshape(x_train, (-1, cifar10_img_size*3)) / 255.0
x_test = reshape(x_test, (-1, cifar10_img_size*3)) / 255.0


linear_model = Sequential()
linear_model.add(Dense(10, activation=sigmoid,
                       input_dim=cifar10_img_size*3))

linear_model.compile(sgd(lr=0.5), mse, metrics=[categorical_accuracy])
linear_model.fit(x_train, y_train,
                 batch_size=8192,
                 epochs=1024,
                 callbacks=[tb_callback],
                 validation_data=(x_test, y_test))

linear_model.save('./model_save/linear/' + experiment_name + '.h5')
plot_model(linear_model, to_file='./model_plot/linear/' + experiment_name + '.png',
           show_shapes=True, show_layer_names=True)