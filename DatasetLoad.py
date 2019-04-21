from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
import scipy.io   
import random
import cv2
import os
def dataset_load(data_source, data_type):
    if data_source == 'train':
        full_path = '/Users/mmao7/Desktop/final/train_32x32.mat'
    elif data_source == 'test':
        full_path = '/Users/mmao7/Desktop/final/test_32x32.mat'
    elif data_source == 'extra':
        full_path = '/Users/mmao7/Desktop/final/extra_32x32.mat'
    else:
        raise Exception('Failed to identify the data resource: ' + data_source + '! Please use: train, test or extra!')
    data = scipy.io.loadmat(full_path, variable_names=data_type).get(data_type)
    return data.transpose((3,0,1,2)) if data_type == 'X' else data[:,0]


def split_dataset(from_X, from_y, N1, N2=None):
    indices = np.random.permutation(from_X.shape[0]) # reshuffle the indices
    DS1_idx = indices[:N1]
    X1, y1 = from_X[DS1_idx,:,:,:], from_y[DS1_idx]
    if N2:
        DS2_idx = indices[N1:N1+N2]
        X2, y2 = from_X[DS2_idx,:,:,:], from_y[DS2_idx]
        return X1, y1, X2, y2
    return X1, y1

def reshuffle_dataset(from_X, from_y):
    indices = np.random.permutation(from_X.shape[0])
    new_X, new_y = from_X[indices], from_y[indices]
    return new_X, new_y

def create_negSample(size, N):
    data_dir = os.path.join('test')
    imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(".png")]
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f))) for f in imagesFiles]
    imgs = [x[:size,:size].copy() for x in imgs]
    imgs = np.array([cv2.resize(x, (size, size)) for x in imgs])
    return imgs[:N,:]

def center_and_normalize(img_dataset):
    imgray_dataset = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_dataset])
    mean = np.mean(imgray_dataset, axis=(1,2), dtype=float)
    std = np.std(imgray_dataset, axis=(1,2), dtype=float, ddof=1)
    imgCN_dataset = np.zeros(imgray_dataset.shape, dtype=float)
    
    for i in np.arange(imgray_dataset.shape[0]):
        if std[i] != 0:
            imgCN_dataset[i,:,:] = (imgray_dataset[i,:,:] - mean[i]) / std[i]
    return imgCN_dataset

def reshape_dataset(dataset, labels):
    dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
    labels = (np.arange(11) == labels[:,None]).astype(np.float32)
    return dataset, labels

def resize_data(X, y, size=(224, 224)):
    X_resized = np.array([cv2.resize(img, size) for img in X])
    y_resized = np.concatenate((np.ones([len(y),989]), y), axis=1)
    return X_resized, y_resized

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

def pickle_store(file_name, all_data):
    try:
      f = open(file_name, 'wb')
      pickle.dump(all_data, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception:
      print('Unable to save data to', file_name, '!')
      raise

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


if __name__ == "__main__":
    tag = '224' # 32
    # load dataset
    train_data = dataset_load('train', 'X')   # shape: (73257, 32, 32, 3) 
    train_labels = dataset_load('train', 'y') # shape: (73257,)

    test_data = dataset_load('test', 'X')   # shape: (26032, 32, 32, 3) 
    test_labels = dataset_load('test', 'y') # shape: (26032,) 

    extra_data = dataset_load('extra', 'X')   # shape: (531131, 32, 32, 3) 
    extra_labels = dataset_load('extra', 'y') # shape: (531131,)
    
    # split some data from train dataset for valid set:
    # train_pos shape: (10000, 32, 32, 3) (10000,)
    # valid_pos shape: (6000, 32, 32, 3) (6000,)
    train_X_pos, train_y_pos, valid_X, valid_y = split_dataset(train_data, train_labels, 10000, 6000)
    # split some data from test dataset for valid set
    # test_pos shape: (6000, 32, 32, 3) (6000,)
    test_X, test_y = split_dataset(test_data, test_labels, 6000)
    
    train_X_neg = create_negSample(32, 8000)     # shape: (8000, 32, 32, 3)
    train_y_neg = np.zeros([8000, ], np.float32) # shape: (8000,)  
    # generate validation set by combining splitted data
    train_X = np.concatenate((train_X_pos, train_X_neg), axis=0)
    train_y = np.concatenate((train_y_pos, train_y_neg), axis=0)
    train_X, train_y = reshuffle_dataset(train_X, train_y)
    print('Training set', train_X.shape, train_y.shape)
    print('Validation set', valid_X.shape, valid_y.shape)
    print('Test set', test_X.shape, test_y.shape)
    if tag == '224':
        train_X, train_y, valid_X, valid_y, test_X, test_y = simple_preprocess(train_X, train_y, valid_X, valid_y, test_X, test_y)
        _train_X, train_y = reshape_dataset(train_X, train_y) #(18000, 32, 32, 3) (18000,11)
        _valid_X, valid_y = reshape_dataset(valid_X, valid_y) # (6000, 32, 32, 3) (6000,11)
        _test_X, test_y = reshape_dataset(test_X, test_y)     # (6000, 32, 32, 3) (6000,11)
        # save data as pickle file
        save = {'train_X': train_X, 'train_y': train_y, 'valid_X': valid_X, 'valid_y': valid_y, 'test_X': test_X, 'test_y': test_y}
        pickle_store('all_data1000.pickle', save)
        # load data from a pickle file
        train_X, train_y, valid_X, valid_y, test_X, test_y = pickle_load('all_data1000.pickle')
        '''
        train_X, train_y = resize_data(train_X, train_y)
        valid_X, valid_y = resize_data(valid_X, valid_y)
        test_X, test_y = resize_data(test_X, test_y)
        print('Training set', train_X.shape, train_y.shape)
        print('Validation set', valid_X.shape, valid_y.shape)
        print('Test set', test_X.shape, test_y.shape)
        
        # save data as pickle file
        save = {'train_X': train_X, 'train_y': train_y, 'valid_X': valid_X, 'valid_y': valid_y, 'test_X': test_X, 'test_y': test_y}
        pickle_store('all_data_ready1000.pickle', save)
        # load data from a pickle file
        train_X, train_y, valid_X, valid_y, test_X, test_y = pickle_load('all_data_ready1000.pickle')
        '''
    else:
        
        '''
        # save data as pickle file
        save = {'train_X': train_X, 'train_y': train_y, 'valid_X': valid_X, 'valid_y': valid_y, 'test_X': test_X, 'test_y': test_y}
        pickle_store('all_data.pickle', save)
        # load data from a pickle file
        train_X, train_y, valid_X, valid_y, test_X, test_y = pickle_load('all_data.pickle')
        '''
        # grayscale, center and normalize 
        train_X = center_and_normalize(train_X)
        valid_X = center_and_normalize(valid_X)
        test_X = center_and_normalize(test_X)
        print('Training set', train_X.shape, train_y.shape)
        print('Validation set', valid_X.shape, valid_y.shape)
        print('Test set', test_X.shape, test_y.shape)
        '''
        # save data as pickle file
        save = {'train_X': train_X, 'train_y': train_y, 'valid_X': valid_X, 'valid_y': valid_y, 'test_X': test_X, 'test_y': test_y}
        pickle_store('all_data_CN.pickle', save)
        # load data from a pickle file
        train_X, train_y, valid_X, valid_y, test_X, test_y = pickle_load('all_data_CN.pickle')
        '''
        train_X, train_y = reshape_dataset(train_X, train_y)
        valid_X, valid_y = reshape_dataset(valid_X, valid_y)
        test_X, test_y = reshape_dataset(test_X, test_y)
        
        # save data as pickle file
        save = {'train_X': train_X, 'train_y': train_y, 'valid_X': valid_X, 'valid_y': valid_y, 'test_X': test_X, 'test_y': test_y}
        pickle_store('all_data_ready.pickle', save)
        # load data from a pickle file
        train_X, train_y, valid_X, valid_y, test_X, test_y = pickle_load('all_data_ready.pickle')
        
    
