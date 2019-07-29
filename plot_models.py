#!/usr/bin/env python3
# -*- coding: utf-8 -*-


""" Plot the Models to png """

from os import listdir
from keras.models import load_model
from keras.utils.vis_utils import plot_model


for file in listdir('model_save'):
    model = load_model('model_save/' + file)
    print('Plotting :', file)
    plot_model(model, to_file='model_plot/' + file + '.png',
               show_shapes=True, show_layer_names=True)
