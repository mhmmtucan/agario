import cv2
import tflearn

import numpy as np
import tensorflow as tf

from utils import Config

from tflearn.layers.conv import conv_2d
from tflearn.layers.conv import max_pool_2d

from tflearn.layers.core import input_data
from tflearn.layers.core import dropout
from tflearn.layers.core import fully_connected

from tflearn.layers.estimator import regression

from tflearn.layers.normalization import batch_normalization
from tflearn.layers.normalization import local_response_normalization

# tensorboard --logdir=foo:C:/Users/H/Desktop/ai-gaming/log

def convnet(config):
    network = input_data(shape=[None, config.sample_height, config.sample_width, config.sample_depth], name='input') # h = 90, w = 150, d = 1

    side_network = input_data(shape=[None, 16], name='extra-input')
    side_network = fully_connected(side_network, 16, activation='relu')
    side_network = tf.tile(side_network, [1, 3375])
    side_network = tf.reshape(side_network, shape=[-1, 45, 75, 16])
    # not sure if we are doing this right

    #side_network1 = input_data(shape=[None, 5], name='prev') # previous areas
    #side_network2 = input_data(shape=[None, 1], name='curr') # current area
    #side_network3 = input_data(shape=[None, 9], name='mous') # mouse position
    #side_network4 = input_data(shape=[None, 1], name='spac') # space hit

    # output_width = ceil(width / stride_width)
    # output_height = ceil(height / stride_height)
    
    network = conv_2d(network, 64, 5, activation='relu') # h = 90, w = 150, d = 64
    network = batch_normalization(network)
    network = conv_2d(network, 64, 5, activation='tanh') # h = 90, w = 150, d = 64
    network = batch_normalization(network)
    network = max_pool_2d(network, 3, strides=2) # h = 45, w = 75, d = 64 
    #network = tf.add(network, side_network);

    network = conv_2d(network, 64, 3, activation='relu') # h = 45, w = 75, d = 64
    network = batch_normalization(network)
    network = conv_2d(network, 64, 3, activation='tanh') # h = 45, w = 75, d = 64
    network = batch_normalization(network)
    network = max_pool_2d(network, 3, strides=2) # h = 23, w = 38, d = 64

    network = conv_2d(network, 64, 3, activation='relu') # h = 23, w = 38, d = 64
    network = batch_normalization(network)
    network = conv_2d(network, 64, 3, activation='tanh') # h = 23, w = 38, d = 64
    network = batch_normalization(network)
    network = max_pool_2d(network, 3, strides=2) # h = 12, w = 19, d = 64

    # this kind of concatination does not look good
    # try to change it
    # try to find better options
    # maybe they can be used as weight or bias
    # did not liked this approach, look into convolutional layers weight and biases to try to use them as weight and bias
    # or look at the multi layer perceptron might be good idea to used those
    #network = tf.concat([network, side_network1, side_network2, side_network3, side_network4], axis=3)
    # since we are doing this way the loss and accuracy will not work properly

    network = fully_connected(network, 100, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 50, activation='tanh')
    network = dropout(network, 0.5)

    #network = fully_connected(network, config.actions, activation='softmax') # frame -> mouse
    network = fully_connected(network, 5, activation='tanh') # frame -> future area

    # need to write custom loss function, categorical crossentropy is good for binary classification
    # mean_squared_error or mean_pairwise_squared_error
    network = regression(network, optimizer='adam', loss=tf.losses.mean_squared_error, learning_rate=1e-3, name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_convnet', max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

    return model

def train_convnet(model, input, test, config):
    X = np.array([i[5] for i in input]).reshape(-1, config.sample_height, config.sample_width, config.sample_depth) # frame
    #X1 = np.array([np.tile(i[0], (12, 19)) for i in input]).reshape(-1, 12, 19, 5) # previous areas
    #X2 = np.array([np.tile(i[1], (12, 19)) for i in input]).reshape(-1, 12, 19, 1) # current area
    #X3 = np.array([np.tile(i[3], (12, 19)) for i in input]).reshape(-1, 12, 19, 9) # mouse position
    #X4 = np.array([np.tile(i[4], (12, 19)) for i in input]).reshape(-1, 12, 19, 1) # space hit
    X1 = [i[0] for i in input] # previous areas
    X2 = [i[1] for i in input] # current area
    X3 = [i[3] for i in input] # mouse position
    X4 = [i[4] for i in input] # space hit
    XSide = np.concatenate((X1, X2, X3, X4), axis=1)
    
    Y = [i[2] for i in input] # future areas

    test_X = np.array([i[5] for i in test]).reshape(-1, config.sample_height, config.sample_width, config.sample_depth)
    #test_X1 = np.array([np.tile(i[0], (12, 19)) for i in test]).reshape(-1, 12, 19, 5) # previous areas
    #test_X2 = np.array([np.tile(i[1], (12, 19)) for i in test]).reshape(-1, 12, 19, 1) # current area
    #test_X3 = np.array([np.tile(i[3], (12, 19)) for i in test]).reshape(-1, 12, 19, 9) # mouse position
    #test_X4 = np.array([np.tile(i[4], (12, 19)) for i in test]).reshape(-1, 12, 19, 1) # space hit
    test_X1 = [i[0] for i in test] # previous areas
    test_X2 = [i[1] for i in test] # current area
    test_X3 = [i[3] for i in test] # mouse position
    test_X4 = [i[4] for i in test] # space hit
    test_XSide = np.concatenate((test_X1, test_X2, test_X3, test_X4), axis=1)

    test_Y = [i[2] for i in test] # future areas

    # start using less epoch since we are going to use a lot of data
    model.fit([X, XSide], Y, n_epoch=6, validation_set=([test_X, test_XSide], test_Y), snapshot_step=500, show_metric=True, run_id='agario-convnet.model')
    #model.fit([X, X1, X2, X3, X4], Y, n_epoch=1, validation_set=([test_X, test_X1, test_X2, test_X3, test_X4], test_Y), snapshot_step=500, show_metric=True, run_id='agario-convnet.model')
    
    model.save('agario-convnet.model')

def show_images(input, test, config):
    input_frames = np.array([i[5] for i in input]).reshape(-1, config.sample_height, config.sample_width, config.sample_depth)
    test_frames = np.array([i[5] for i in test]).reshape(-1, config.sample_height, config.sample_width, config.sample_depth)

    for frame in input_frames:
        cv2.imshow('input', frame)
        cv2.waitKey(1)

    cv2.destroyWindow('input')

    for frame in test_frames:
        cv2.imshow('test', frame)
        cv2.waitKey(1)

    cv2.destroyWindow('test')

def create_model(filename):
    config = Config()
    
    train_data = np.load(filename)
    test_data_len = len(train_data) // 10
    #print(test_data_len)
    input = train_data[:-test_data_len]
    test = train_data[-test_data_len:]
    print('got the data')

    #show_images(input, test, config)

    tf.get_default_graph()
    tf.reset_default_graph()
    tf.get_default_graph()

    model = convnet(config)
    print('created the model')

    train_convnet(model, input, test, config)
    print('trained the model')