import os
import keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt

import malticlass_cnn
import image_generator

class ExperimentCifar10:
    def __init__(self):
        self.NUM_CLASS = 10
        self.INPUT_SHAPE = (32, 32, 3)
        return

    def __get_cifar10_data(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        #
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train = (x_train - 127.5) / 127.5
        x_test = (x_test - 127.5) / 127.5

        #
        y_train = keras.utils.to_categorical(y_train, self.NUM_CLASS)
        y_test = keras.utils.to_categorical(y_test, self.NUM_CLASS)

        return x_train, y_train, x_test, y_test

    def run_resnetv1(self, drop_act_rate, csvlog_file):
        # data
        x_train, y_train, x_test, y_test = self.__get_cifar10_data()

        # model
        resnetv1 = malticlass_cnn.ResNet_v1(self.NUM_CLASS, self.INPUT_SHAPE, depth=20, drop_act_rate=drop_act_rate)
        #
        resnetv1.built_model()
        #
        EPOCH = 200
        BATCH_SIZE = 256
        LEARNING_RATE = 0.001
        CSVLOG_FILE = csvlog_file
        resnetv1.train_model(x_train, y_train, x_test, y_test, 
                        EPOCH, BATCH_SIZE, 
                        LEARNING_RATE,
                        CSVLOG_FILE)
        return

    def run_resnetv1_withDA(self, drop_act_rate, csvlog_file, random_erase=False, mixup=False, cutout=False):
        # data
        x_train, y_train, x_test, y_test = self.__get_cifar10_data()

        # image generator
        IMAGE_GEN_KWARGS = {
            # set input mean to 0 over the dataset
            "featurewise_center":False,
            # set each sample mean to 0
            "samplewise_center":False,
            # divide inputs by std of dataset
            "featurewise_std_normalization":False,
            # divide each input by its std
            "samplewise_std_normalization":False,
            # apply ZCA whitening
            "zca_whitening":False,
            # epsilon for ZCA whitening
            "zca_epsilon":1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            "rotation_range":0,
            # randomly shift images horizontally
            "width_shift_range":0.1,
            # randomly shift images vertically
            "height_shift_range":0.1,
            # set range for random shear
            "shear_range":0.,
            # set range for random zoom
            "zoom_range":0.,
            # set range for random channel shifts
            "channel_shift_range":0.,
            # set mode for filling points outside the input boundaries
            "fill_mode":'nearest',
            # value used for fill_mode : "constant"
            "cval":0.,
            # randomly flip images
            "horizontal_flip":True,
            # randomly flip images
            "vertical_flip":False,
            # set rescaling factor (applied before any other transformation)
            "rescale":None,
            # set function that will be applied on each input
            "preprocessing_function":None,
            # image data format, either "channels_first" or "channels_last"
            "data_format":None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            "validation_split":0.0}
        if random_erase:
            RANDOM_ERASING_KWARGS = {'erasing_prob':0.5, 
                                 'area_rate_low':0.02, 
                                 'area_rate_high':0.4, 
                                 'aspect_rate_low':0.3, 
                                 'aspect_rate_high':3.3}
        else:
            RANDOM_ERASING_KWARGS = None

        if mixup:
            MIXUP_ALPHA = 1
        else:
            MIXUP_ALPHA = None

        if cutout:
            CUTOUT_KWARGS = {'num_holes':1, 
                             'max_h_size':16, 
                             'max_w_size':16, 
                             'always_apply':False, 
                             'p':0.3}
        else:
            CUTOUT_KWARGS = None

        # model
        resnetv1 = malticlass_cnn.ResNet_v1(self.NUM_CLASS, self.INPUT_SHAPE, depth=20, drop_act_rate=drop_act_rate)
        #
        resnetv1.built_model()
        #
        EPOCH = 200
        BATCH_SIZE = 256
        LEARNING_RATE = 0.001
        CSVLOG_FILE = csvlog_file
        #
        mydatagene = image_generator.MyImageDataGenerator(x_train,
                                                         y_train,
                                                         BATCH_SIZE, 
                                                         IMAGE_GEN_KWARGS,
                                                         random_erasing_kwargs=RANDOM_ERASING_KWARGS,
                                                         mixup_alpha=MIXUP_ALPHA,
                                                         cutout_kwargs=CUTOUT_KWARGS,
                                                         )
        #imgs, ys = mydatagene[0]
        #for img in imgs:
        #    plt.imshow(img)
        #    plt.show()
        
        #
        resnetv1.train_model_with_datagene(mydatagene, x_test, y_test, 
                        EPOCH,
                        LEARNING_RATE,
                        CSVLOG_FILE)
        return


#ExperimentCifar10().run_resnetv1(0.5, 'log_dropAct050.csv')
#ExperimentCifar10().run_resnetv1_withDA(0.00, 'log_dropAct000_cutout.csv', cutout=True)
ExperimentCifar10().run_resnetv1_withDA(0.05, 'log_dropAct005_cutout.csv', cutout=True)
ExperimentCifar10().run_resnetv1_withDA(0.00, 'log_dropAct000_randomerase.csv', random_erase=True)
ExperimentCifar10().run_resnetv1_withDA(0.05, 'log_dropAct005_randomerase.csv', random_erase=True)
ExperimentCifar10().run_resnetv1_withDA(0.00, 'log_dropAct000_mixup.csv', mixup=True)
ExperimentCifar10().run_resnetv1_withDA(0.05, 'log_dropAct005_mixup.csv', mixup=True)
ExperimentCifar10().run_resnetv1_withDA(0.00, 'log_dropAct000_mixup_cutout.csv', mixup=True, cutout=True)
ExperimentCifar10().run_resnetv1_withDA(0.05, 'log_dropAct005_mixup_cutout.csv', mixup=True, cutout=True)
ExperimentCifar10().run_resnetv1_withDA(0.00, 'log_dropAct000_mixup_randomerase.csv', mixup=True, random_erase=True)
ExperimentCifar10().run_resnetv1_withDA(0.05, 'log_dropAct005_mixup_randomerase.csv', mixup=True, random_erase=True)
