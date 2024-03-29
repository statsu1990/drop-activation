"""
https://keras.io/examples/cifar10_resnet/
"""

import os
import keras
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, CSVLogger
from keras.regularizers import l2
from keras.utils import plot_model
import numpy as np

from drop_activation import DropActivation

class ResNet_v1:
    def __init__(self, class_num, input_shape, depth, drop_act_rate):
        self.CLASS_NUM = class_num
        self.INPUT_SHAPE = input_shape
        self.DEPTH = depth
        self.DROP_ACT_RATE = drop_act_rate

        return

    def built_model(self):
        self.model = self.resnet_v1(self.DEPTH)
        return

    def train_model(self, x_train, y_train, x_test, y_test, 
                    epochs, batch_size, 
                    learning_rate,
                    csvlog_file):
        
        # compile
        self.model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])

        # summary
        self.model.summary()

        # call back
        callbacks = []
        # lr schedule
        self.LEARNING_RATE = learning_rate
        lr_scheduler = LearningRateScheduler(self.__lr_schedule)
        callbacks.append(lr_scheduler)
        # csv logger
        csvlogger = CSVLogger(csvlog_file)
        callbacks.append(csvlogger)

        # datagen
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
        datagen.fit(x_train)

        # fit
        self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            epochs=epochs,
                            steps_per_epoch=int(x_train.shape[0] / batch_size),
                            validation_data=(x_test, y_test),
                            callbacks=callbacks,
                            )

        # score
        scores = self.model.evaluate(x_train, y_train, verbose=0)
        print('Train loss:', scores[0])
        print('Train accuracy:', scores[1])

        scores = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        return


    def train_model_with_datagene(self, train_datagene, x_test, y_test, 
                    epochs, 
                    learning_rate,
                    csvlog_file,):
        
        # compile
        self.model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])

        # summary
        self.model.summary()

        # call back
        callbacks = []
        # lr schedule
        self.LEARNING_RATE = learning_rate
        lr_scheduler = LearningRateScheduler(self.__lr_schedule)
        callbacks.append(lr_scheduler)
        # csv logger
        csvlogger = CSVLogger(csvlog_file)
        callbacks.append(csvlogger)

        # fit
        self.model.fit_generator(train_datagene,
                            epochs=epochs,
                            steps_per_epoch=len(train_datagene),
                            validation_data=(x_test, y_test),
                            callbacks=callbacks,
                            )

        # score
        #scores = self.model.evaluate(x_train, y_train, verbose=0)
        #print('Train loss:', scores[0])
        #print('Train accuracy:', scores[1])

        scores = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        return

    def resnet_layer(self, inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder

        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)

        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                #x = Activation(activation)(x)
                x = self.__relu_layer(drop_act_rate=self.DROP_ACT_RATE)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                #x = Activation(activation)(x)
                x = self.__relu_layer(drop_act_rate=self.DROP_ACT_RATE)(x)
            x = conv(x)
        return x

    def resnet_v1(self, depth):
        """
        https://keras.io/examples/cifar10_resnet/
        
        ResNet Version 1 Model builder [a]

        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = Input(shape=self.INPUT_SHAPE)
        x = self.resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = self.resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = self.resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(self.CLASS_NUM,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def __relu_layer(self, drop_act_rate=None):
        if drop_act_rate is None:
            return Activation('relu')
        else:
            return DropActivation(rate=drop_act_rate)

    def __lr_schedule(self, epoch):
        """Learning Rate Schedule

        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        lr = self.LEARNING_RATE
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        return lr