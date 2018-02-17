import cv2
import tflearn

import numpy as np
import tensorflow as tf

from utils import Config

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

# tensorboard --logdir=foo:C:/Users/H/Desktop/ai-gaming/log

def alexnet(config):
    network = input_data(shape=[None, config.sample_height, config.sample_width, config.sample_depth], name='input')

    side_network1 = input_data(shape=[None, 5], name='prev') # previous areas
    side_network2 = input_data(shape=[None, 1], name='curr') # current area
    side_network3 = input_data(shape=[None, 9], name='mous') # mouse position
    side_network4 = input_data(shape=[None, 1], name='spac') # space hit

    network = conv_2d(network, 64, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 32, 7, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 16, 5, activation='relu')
    network = conv_2d(network, 16, 5, activation='relu')
    network = conv_2d(network, 16, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 100, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 50, activation='tanh')
    network = dropout(network, 0.5)

    #network = fully_connected(network, config.actions, activation='softmax') # frame -> mouse
    network = fully_connected(network, 5, activation='softmax') # frame -> future area

    # need to write custom loss function, categorical crossentropy is good for binary classification
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=1e-3, name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_alexnet', max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

    return model

def train_alexnet(model, input, test, config):
    X = np.array([i[5] for i in input]).reshape(-1, config.sample_height, config.sample_width, config.sample_depth) # frame
    X1 = [i[0] for i in input] # previous areas
    X2 = [i[1] for i in input] # current area
    X3 = [i[3] for i in input] # mouse position
    X4 = [i[4] for i in input] # space hit

    Y = [i[2] for i in input] # future areas

    test_X = np.array([i[5] for i in test]).reshape(-1, config.sample_height, config.sample_width, config.sample_depth)
    test_X1 = [i[0] for i in test] # previous areas
    test_X2 = [i[1] for i in test] # current area
    test_X3 = [i[3] for i in test] # mouse position
    test_X4 = [i[4] for i in test] # space hit

    test_Y = [i[2] for i in test] # future areas

    model.fit([X, X1, X2, X3, X4], Y, n_epoch=8, validation_set=([test_X, test_X1, test_X2, test_X3, test_X4], test_Y), snapshot_step=500, show_metric=True, run_id='agario-alexnet.model')
    
    model.save('agario-alexnet.model')

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

if __name__ == '__main__':
    config = Config()
    
    train_data = np.load('training_data.npy')
    test_data_len = len(train_data) // 10
    #print(test_data_len)
    input = train_data[:-test_data_len]
    test = train_data[-test_data_len:]
    print('got the data')

    #show_images(input, test, config)

    tf.get_default_graph()
    tf.reset_default_graph()
    tf.get_default_graph()

    model = alexnet(config)
    print('created the model')

    train_alexnet(model, input, test, config)
    print('trained the model')