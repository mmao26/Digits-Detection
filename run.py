# import the necessary packages
from __future__ import absolute_import
from __future__ import print_function

import os
import itertools
import numpy as np
import scipy.io
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


session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import cv2
import numpy as np
import time
import os
import scipy.signal

class myCNN:
# WITH Batch Normalization
    def __init__(self):
        self.epoch = 30
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


    def load_model(self, img, threshold):
        json_file = open('model_pos.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model_pos.h5")
        #print("Loaded model from disk")

        #loaded_model.compile(loss='categorical_crossentropy', optimizer=self.sgd, metrics=['accuracy'])
        yhat = loaded_model.predict(img.reshape((1, 32, 32, 1)))
        label = loaded_model.predict_classes(img.reshape((1, 32, 32, 1))) # im.reshape((1, 32, 32, 1))
        #print('This time yhat & label: ', np.max(yhat), label[0])
        if np.max(yhat) < threshold or label[0] == 0:
            return '0', yhat
        else:
            return str(label[0]), yhat


def center_and_normalize(x):
    return (x - x.mean()) / x.std()

def draw_label(image_in, box_upright, label):
    """
    Args:
        image_in (numpy.array): input image.
        label (str): label values can be: '0','1', ... '9'.

    Returns:
        numpy.array: output image showing a label.
    """
    image_out = image_in
    text_loc = (box_upright[0], box_upright[1])
    cv2.putText(image_out, "{}".format(label),
                text_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image_out

def generatingKernel(a):
    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)


def reduce_layer(image, kernel=generatingKernel(0.4)):
    """Convolve the input image with a generating kernel of parameter of 0.4
    and then reduce its width and height each by a factor of two.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data type
        (e.g., np.uint8, np.float64, etc.)

    kernel : numpy.ndarray (Optional)
        A kernel of shape (N, N). The array may have any data type (e.g.,
        np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        An image of shape (ceil(r/2), ceil(c/2)). For instance, if the input is
        5x7, the output will be 3x4.
    """

    N = kernel.shape[0] // 2
    padded_image = cv2.copyMakeBorder(image, N, N, N, N, borderType=cv2.BORDER_REFLECT)
    filtered_image = scipy.signal.convolve2d(padded_image, kernel, 'same')
    [r,c] = image.shape
    valid_image = np.zeros([r,c])
    for i in range(N, r+N):
        for j in range (N, c+N):
            valid_image[i-N,j-N] = filtered_image[i,j]
    return valid_image[::2, ::2]

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def sliding_window_detector(image_name, winW, winH, stepSize, thre, min_i, max_len):

    agentMyCNN = myCNN()

    img_list = []
    label_list = []
    image = cv2.imread(os.path.join('images5', image_name), 0)
    image = center_and_normalize(image)
    origin_image = cv2.imread(os.path.join('images5', image_name))
    cv2.imwrite("graded_images/{}.png".format('haha0'), image)
    print (image.shape)
    temp_image = np.copy(image)
    temp_ori_image = np.copy(origin_image)
    # loop over the image pyramid
    prev_label = '0'
    for i in range(min_i, min_i+1):
        #print ('i now is', i)
        resized_img = reduce_layer(temp_image).copy() if i else temp_image.copy()
        #print ('img size now is', resized_img.shape)
        resized_ori_img = cv2.resize(temp_ori_image.copy(), (0,0), fx=0.5, fy=0.5) if i else temp_ori_image.copy()

        temp_image = np.copy(resized_img)
        temp_ori_image = np.copy(resized_ori_img)
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized_img, stepSize, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                    continue

            window_resized = cv2.resize(window.copy(), (32, 32))
            label, prob = agentMyCNN.load_model(window_resized, thre)
            if label != '0':
                # since we do not have a classifier, we'll just draw the window
                if thre == 0.47:
                    cur_label = label
                    if prev_label == '2' and cur_label == '5':
                        #print('valid prob & label: ', np.max(prob), label)

                        clone = resized_ori_img.copy()
                        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                        clone_out = draw_label(clone, (x + winW, y), label)
                        cv2.imwrite("graded_images/{}.png".format('haha1'), clone_out)
                        img_list.append(clone_out)
                        label_list.append(label)
                        if len(img_list) >= max_len:
                            return img_list, label_list
                    prev_label = cur_label
                else:
                    #print('valid prob & label: ', np.max(prob), label)

                    clone = resized_ori_img.copy()
                    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                    clone_out = draw_label(clone, (x + winW, y), label)
                    cv2.imwrite("graded_images/{}.png".format('haha1'), clone_out)
                    img_list.append(clone_out)
                    label_list.append(label)
                    if len(img_list) >= max_len:
                        return img_list, label_list
    return img_list, label_list

def final_image_generator(img_list, label_list, img_out_name):
    # img_out_name: '1', '2', '3', '4', '5'
    # support 9 digits detection
    if len(img_list) == 1:
        img_temp = img_list[0]
    else:
        ratioList = [(0.5, 0.5), (0.67, 0.33), (0.75, 0.25), (0.8, 0.2), (0.83, 0.17), (0.86, 0.14), (0.875, 0.125), (0.89, 0.11), (0.9, 0.1)]
        for i in range(len(img_list)-1):
            img1 = img_temp.copy() if i else img_list[0].copy()
            img2 = img_list[i+1].copy()
            img_temp = cv2.addWeighted(img1, ratioList[i][0], img2, ratioList[i][1], 0)
    cv2.imwrite("graded_images/{}.png".format(img_out_name), img_temp)


if __name__ == "__main__":
    # load the image and define the window width and height

    img_list1, labels_list1 = sliding_window_detector('IMG14.jpeg', 55, 90, 20, 0.996, 0, 3)
    final_image_generator(img_list1[1:], labels_list1[1:], '1')

    img_list2, labels_list2 = sliding_window_detector('IMG14_G.jpg', 20, 32, 8, 0.53, 1, 11)
    final_image_generator(img_list2[9:11], labels_list2[9:11], '2')

    img_list3, labels_list3 = sliding_window_detector('IMG23.jpg', 25, 40, 10, 0.8, 0, 10)
    final_image_generator(img_list3[8:10], labels_list3[8:10], '3')

    img_list4, labels_list4 = sliding_window_detector('IMG4.jpeg', 64, 64, 16, 0.99, 0, 1)
    final_image_generator(img_list4, labels_list4, '4')

    img_list51, labels_list51 = sliding_window_detector('IMG25.jpeg', 40, 70, 30, 0.85, 0, 1) # 2
    img_list52, labels_list52 = sliding_window_detector('IMG25.jpeg', 40, 70, 20, 0.47, 0, 1) # 5
    img_list5 = [img_list51[0], img_list52[0]]
    labels_list5 = [labels_list51[0], labels_list52[0]]
    final_image_generator(img_list5, labels_list5, '5')
