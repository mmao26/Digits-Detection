from __future__ import absolute_import
from __future__ import print_function

import os
import itertools
import numpy as np
import scipy.io
from six.moves import cPickle as pickle
import tensorflow as tf

import keras
from keras.models import Sequential, Model, load_model
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.models import model_from_json
from keras import applications
from keras import optimizers

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import cv2

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def pickle_load(file_name):
    with open(file_name, 'rb') as f:
        all_data = pickle.load(f)
        train_X = all_data['train_X']
        train_y = all_data['train_y']
        valid_X = all_data['valid_X']
        valid_y = all_data['valid_y']
        test_X = all_data['test_X']
        test_y = all_data['test_y']
        del all_data  # release memory
        print('Training set', train_X.shape, train_y.shape)
        print('Validation set', valid_X.shape, valid_y.shape)
        print('Test set', test_X.shape, test_y.shape)
        return train_X, train_y, valid_X, valid_y, test_X, test_y

def resize_data(dataset, size=(224, 224)):
    dataset_resized = np.array([cv2.resize(img, size) for img in dataset])
    return dataset_resized

def simple_preprocess(train_X, train_y, valid_X, valid_y, test_X, test_y):
    train_X = train_X.astype('float32')
    train_y = train_y.astype('float32')
    train_X /= 255
    train_y /= 255
    valid_X = valid_X.astype('float32')
    valid_y = valid_y.astype('float32')
    valid_X /= 255
    valid_y /= 255
    test_X = test_X.astype('float32')
    test_y = test_y.astype('float32')
    test_X /= 255
    test_y /= 255
    return train_X, train_y, valid_X, valid_y, test_X, test_y

class myCNN:
# WITH Batch Normalization
    def __init__(self):
        self.batch_size = 1024
        self.epoch = 20
        self.momentum = 0.9
        self.decay = 1e-6
        self.learning_rate = 0.01
        self.num_class = 11
        self.model = self._build_model()


    def _build_model(self):
        model = Sequential()
        # Layer 1: conv3-32
        model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(32, 32, 1)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # Layer 2: conv3-32
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # Layer 3: conv3-64
        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # Layer 4: conv3-64
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # FC Layer
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # FC Layer
        model.add(Dense(self.num_class))
        model.add(Activation('softmax'))
        self.sgd = SGD(lr=self.learning_rate, decay=self.decay, momentum=self.momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=self.sgd, metrics=['accuracy'])
        return model

    def train(self, train_X, train_y, valid_X, valid_y):
        history = self.model.fit(train_X, train_y, batch_size=self.batch_size, epochs=self.epoch, verbose=2, validation_data=(valid_X, valid_y))
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'], marker='8')
        plt.plot(history.history['val_acc'], marker='8')
        plt.title('Model Accuracy (%)', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.xlabel('Epoch', fontsize=14)
        plt.legend(['Train', 'Validation'], loc='lower right', fontsize=13)
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'], marker='8')
        plt.plot(history.history['val_loss'], marker='8')
        plt.title('Model Loss', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.xlabel('Epoch', fontsize=14)
        plt.legend(['Train', 'Validation'], loc='upper right', fontsize=13)
        plt.show()

    def predict(self, test_X, test_y):
        score = self.model.evaluate(test_X, test_y, verbose=0)
        print('loss:', score[0])
        print('Test accuracy:', score[1])

    def save_model(self):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

    def load_model(self, evl_flag, test_X, test_y):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        if evl_flag:
            loaded_model.compile(loss='categorical_crossentropy', optimizer=self.sgd, metrics=['accuracy'])
            score = loaded_model.evaluate(test_X, test_y, verbose=0)
            print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

class VGG:
    def __init__(self):
        self.batch_size = 1024
        self.epoch = 20
        self.momentum = 0.9
        self.decay = 1e-6
        self.learning_rate = 0.01
        self.num_class = 11
        self.steps_per_epoch = 1
        self.validation_steps = 1
        self.model = self._build_model()


    def _build_model(self):
        model = Sequential()
        # Layer 1: conv3-64
        model.add(Convolution2D(64, 3, 3, border_mode='same',input_shape=(32, 32, 1)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # Layer 2: conv3-64
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # Layer 3: conv3-128
        model.add(Convolution2D(128, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # Layer 4: conv3-128
        model.add(Convolution2D(128, 3, 3))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # Layer 5: conv3-256
        model.add(Convolution2D(256, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # Layer 6: conv3-256
        model.add(Convolution2D(256, 3, 3))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # Layer 7: conv3-512
        model.add(Convolution2D(512, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # FC Layer
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # FC Layer
        model.add(Dense(self.num_class))
        model.add(Activation('softmax'))
        self.sgd = SGD(lr=self.learning_rate, decay=self.decay, momentum=self.momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=self.sgd, metrics=['accuracy'])
        return model


    def train(self, train_X, train_y, valid_X, valid_y):
        history = self.model.fit(train_X, train_y, epochs=self.epoch, verbose=2, validation_data=(valid_X, valid_y))
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'], marker='8')
        plt.plot(history.history['val_acc'], marker='8')
        plt.title('Model Accuracy (%)', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.xlabel('Epoch', fontsize=14)
        plt.legend(['Train', 'Validation'], loc='lower right', fontsize=13)
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'], marker='8')
        plt.plot(history.history['val_loss'], marker='8')
        plt.title('Model Loss', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.xlabel('Epoch', fontsize=14)
        plt.legend(['Train', 'Validation'], loc='upper right', fontsize=13)
        plt.show()

    def predict(self, test_X, test_y):
        score = self.model.evaluate(test_X, test_y, verbose=0)
        print('loss:', score[0])
        print('Test accuracy:', score[1])

    def save_model(self):
        model_json = self.model.to_json()
        with open("model_vgg16.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model_vgg16.h5")
        print("Saved model to disk")

    def load_model(self, evl_flag, test_X, test_y):
        json_file = open('model_vgg16.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model_vgg16.h5")
        print("Loaded model from disk")
        if evl_flag:
            loaded_model.compile(loss='categorical_crossentropy', optimizer=self.sgd, metrics=['accuracy'])
            score = loaded_model.evaluate(test_X, test_y, verbose=0)
            print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model

def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


if __name__ == "__main__":
    runfile = 'VGG16_pretrained' # 'VGG' 'VGG16_pretrained'

    if runfile == 'myCNN':
        train_X, train_y, valid_X, valid_y, test_X, test_y = pickle_load('all_data_ready.pickle')
        print('Training set', train_X.shape, train_y.shape)
        print('Validation set', valid_X.shape, valid_y.shape)
        print('Test set', test_X.shape, test_y.shape)
        agentMyCNN = myCNN()
        agentMyCNN.train(train_X, train_y, valid_X, valid_y)
        agentMyCNN.predict(test_X, test_y)
        agentMyCNN.save_model()
        agentMyCNN.load_model(1, test_X, test_y)
        #Test accuracy: 91.8%
    elif runfile == 'VGG':
        train_X, train_y, valid_X, valid_y, test_X, test_y = pickle_load('all_data_ready.pickle')
        agentVGG = VGG()
        agentVGG.train(train_X, train_y, valid_X, valid_y)
        agentVGG.predict(test_X, test_y)
        agentVGG.save_model()
        agentVGG.load_model(1, test_X, test_y)
        #Test accuracy: 94.8%
    elif runfile == 'VGG16_pretrained':
        '''
        train_X, train_y, valid_X, valid_y, test_X, test_y = pickle_load('all_data_ready.pickle')
        agentVGG16pre = VGG16_Pre()
        agentVGG16pre.train(train_X, train_y, valid_X, valid_y)
        agentVGG16pre.predict(test_X, test_y)
        agentVGG16pre.save_model()
        agentVGG16pre.load_model(1, test_X, test_y)
        '''
        train_X, train_y, valid_X, valid_y, test_X, test_y = pickle_load('all_data_ready1000.pickle')
        pretrained_model = vgg11_bn(pretrained=True)
        pretrained_model.predict(test_X, test_y)
        #Test accuracy: 94.2%
