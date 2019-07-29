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
from keras.activations import relu, softmax #, sigmoid
from keras.metrics import categorical_accuracy, categorical_crossentropy #, mse
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout


experiment_name = 'CIPHAR10_DEEPCONVNET_3_128_CONV_MAXPOOL_33_256DENSE_SM_LR_05_DCY_1E-4_BN_DROP_LOSS_CCE'
tb_callback = TensorBoard('./logs/' + experiment_name)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Transformation to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2D Convolution
x_train = reshape(x_train, (-1, 32, 32, 3)) / 255.0
x_test = reshape(x_test, (-1, 32, 32, 3)) / 255.0

inputs = Input((32, 32, 3))
dropout = Dropout(rate=0.2)(inputs)
hidden_conv = Conv2D(128, (3, 3), activation=relu, padding='same')(dropout)
batch_normalization = BatchNormalization()(hidden_conv)
dropout = Dropout(rate=0.3)(hidden_conv)
pool_layers = MaxPooling2D(pool_size=(2, 2))(dropout)

for i in range(2):
    # dr_rate = (0.2 + (i / (2 * 10)))
    hidden_conv = Conv2D(128, (3, 3), activation=relu, padding='same')(pool_layers)
    batch_normalization = BatchNormalization()(hidden_conv)
    pool_layers = MaxPooling2D(pool_size=(2, 2))(batch_normalization)
    dropout = Dropout(rate=0.3)(pool_layers)

flatten = Flatten()(dropout)
hidden_dense = Dense(256, activation=relu)(flatten)
batch_normalization = BatchNormalization()(hidden_dense)
dropout = Dropout(rate=0.4)(batch_normalization)
outputs = Dense(10, activation=softmax)(dropout)

model = Model(inputs, outputs)
sgd_opt = sgd(lr=0.5, decay=1e-4)
model.compile(optimizer=sgd_opt, loss=categorical_crossentropy,
              metrics=[categorical_accuracy])
model.fit(x_train, y_train,
          batch_size=256,
          epochs=120,
          callbacks=[tb_callback],
          validation_data=(x_test, y_test))

model.save('./model_save/deep_convnet/' + experiment_name + '.h5')
plot_model(model, to_file='./model_plot/deep_convnet/' + experiment_name + '.png',
           show_shapes=True, show_layer_names=True)
