#!/usr/bin/env python3
# -*- coding: utf-8 -*-


""" Deep Convnet for the CIFAR 10 Project """


# from pprint import pprint
from numpy import reshape
from keras import Model
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import sgd
from keras.callbacks import TensorBoard
from keras.activations import relu, sigmoid
from keras.metrics import categorical_accuracy, mse
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout


experiment_name = """CIPHAR10_DEEPCONVNET_32_64_128_CONV*2_MAXPOOL_22_
256DENSE_LR_01_DCY_1E-6_N_MMT_09_BN_DROP"""
tb_callback = TensorBoard('./logs/' + experiment_name)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Transformation to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2D Convolution
x_train = reshape(x_train, (-1, 32, 32, 3)) / 255.0
x_test = reshape(x_test, (-1, 32, 32, 3)) / 255.0

inputs = Input((32, 32, 3))
# dropout = Dropout(rate=0.2)(inputs)
hidden_conv_1 = Conv2D(32, (3, 3), activation=relu, padding='same')(inputs)
batch_normalization = BatchNormalization()(hidden_conv_1)
dropout = Dropout(rate=0.2)(batch_normalization)
hidden_conv_2 = Conv2D(32, (3, 3), activation=relu, padding='same')(dropout)
batch_normalization = BatchNormalization()(hidden_conv_2)
dropout = Dropout(rate=0.2)(batch_normalization)
pool_layers = MaxPooling2D(pool_size=(2, 2))(dropout)

for i in range(2, 6, 2):
    dr_rate = (0.2 + (i / (2 * 10)))
    hidden_conv_1 = Conv2D(32*i, (3, 3), activation=relu, padding='same')(pool_layers)
    batch_normalization = BatchNormalization()(hidden_conv_1)
    dropout = Dropout(rate=dr_rate)(batch_normalization)
    hidden_conv_2 = Conv2D(32*i, (3, 3), activation=relu, padding='same')(dropout)
    batch_normalization = BatchNormalization()(hidden_conv_2)
    dropout = Dropout(rate=dr_rate)(batch_normalization)
    pool_layers = MaxPooling2D(pool_size=(2, 2))(dropout)

flatten = Flatten()(pool_layers)
hidden_dense = Dense(256, activation=relu)(flatten)
batch_normalization = BatchNormalization()(hidden_dense)
dropout = Dropout(rate=0.4)(batch_normalization)
outputs = Dense(10, activation=sigmoid)(dropout)

model = Model(inputs, outputs)
sgd_opt = sgd(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd_opt, loss=mse, metrics=[categorical_accuracy])
model.fit(x_train, y_train,
          batch_size=256,
          epochs=120,
          callbacks=[tb_callback],
          validation_data=(x_test, y_test))

model.save('./model_save/deep_convnet/' + experiment_name + '.h5')
plot_model(model, to_file='./model_plot/deep_convnet/' + experiment_name + '.png',
           show_shapes=True, show_layer_names=True)
