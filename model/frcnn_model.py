
from __future__ import print_function, division
from keras.datasets import mnist
import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, TimeDistributed,Convolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import  Conv2D
from keras.layers import UpSampling2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.applications import ResNet50
from keras.engine import base_layer
from keras import layers
from keras.applications import ResNet50
import keras.backend as K
import tensorflow as tf

import matplotlib.pyplot as plt
from keras.objectives import categorical_crossentropy
import os
import numpy as np
from tools.base_layers import RoiPoolingConv, classifier_layers


lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

class FRCNN():
    def __init__(self, num_classes, num_anchors):
        self.num_classes=num_classes
        self.num_anchors=num_anchors
    # def rpn_loss_regr(self):
    #     num_anchors=self.num_anchors
    #     def rpn_loss_regr_fixed_num(y_true, y_pred):
    #         if K.image_dim_ordering() == 'th':
    #             x = y_true[:, 4 * num_anchors:, :, :] - y_pred
    #             x_abs = K.abs(x)
    #             x_bool = K.less_equal(x_abs, 1.0)
    #             return lambda_rpn_regr * K.sum(
    #                 y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
    #         else:
    #             x = y_true[:, :, :, 4 * num_anchors:] - y_pred
    #             x_abs = K.abs(x)
    #             x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

    #             return lambda_rpn_regr * K.sum(
    #                 y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

    #     return rpn_loss_regr_fixed_num

    # def rpn_loss_cls(self):
    #     num_anchors=self.num_anchors
    #     def rpn_loss_cls_fixed_num(y_true, y_pred):
    #         return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
    #     return rpn_loss_cls_fixed_num


    # def class_loss_regr(self):
    #     num_classes=self.num_classes-1
    #     def class_loss_regr_fixed_num(y_true, y_pred):
    #         x = y_true[:, :, 4*num_classes:] - y_pred
    #         x_abs = K.abs(x)
    #         x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
    #         return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
    #     return class_loss_regr_fixed_num


    # def class_loss_cls(self):
    #     def class_loss_cls_(y_true, y_pred):
    #         return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
    #     return class_loss_cls_
    def build_share(self):
        input_shape_img = (None, None, 3)

        img_input = Input(shape=input_shape_img)
        model = ResNet50(weights='imagenet',include_top=False,input_tensor=img_input)
        x = model.layers[141].output
        model=Model(img_input,x)
        model.summary()
        return model

    def build_rpn(self,base_layers):
        num_anchors=self.num_anchors
        x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
            base_layers)

        x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                                name='rpn_out_class')(x)
        x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                            name='rpn_out_regress')(x)
        
        return [x_class, x_regr]
    def build_classifier(self,base_layers, input_rois, num_rois=32, nb_classes=21, trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
        nb_classes = self.num_classes
        pooling_regions = 14
        input_shape = (num_rois, 14, 14, 1024)

        out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
        out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

        out = TimeDistributed(Flatten())(out)

        out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                    name='dense_class_{}'.format(nb_classes))(out)
        # note: no regression target for bg class
        out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                                name='dense_regress_{}'.format(nb_classes))(out)
        return [out_class, out_regr]