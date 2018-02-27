import tflearn

import numpy as np
import tensorflow as tf

from random import shuffle

from tflearn.layers.conv import conv_2d
from tflearn.layers.conv import max_pool_2d

from tflearn.layers.core import dropout
from tflearn.layers.core import input_data
from tflearn.layers.core import fully_connected

from tflearn.layers.estimator import regression

from tflearn.layers.normalization import batch_normalization
from tflearn.layers.normalization import local_response_normalization

from tensorflow.python.lib.io import file_io

# tensorboard --logdir=foo:C:/Users/malkoch/Desktop/Github/agario/log
sample_height = 150
sample_width = 90
sample_depth = 1

# Google Storage locations
GS_BUCKET               = "gs://agarflow-ml-storage-1773/"
INPUT_FOLDER            = "input/"
OUTPUT_FOLDER           = "output/"
INPUT_FILENAME          = "combined_raw.npy"
OUTPUT_FILENAME         = "agario-convnet.model"

def read_bucket_np(filename):
    # reads a file from Google Storage
    with file_io.FileIO(GS_BUCKET+INPUT_FOLDER+filename, "r") as file:
        data = np.load(file)
    return data


def write_bucket_np(data, filename):
    # writes a file to Google Storage
    with file_io.FileIO(GS_BUCKET+OUTPUT_FOLDER+filename, "w") as file:
        np.save(file, data)


def convnet():
    network = input_data(shape=[None, sample_height, sample_width, sample_depth],name='input')  # h = 90, w = 150, d = 1

    side_network = input_data(shape=[None, 16], name='extra-input')
    side_network = fully_connected(side_network, 64, activation='relu')
    side_network = tf.tile(side_network, [1, 3375])
    side_network = tf.reshape(side_network, shape=[-1, 45, 75, 64])
    # not sure if we are doing this right

    # output_width = ceil(width / stride_width)
    # output_height = ceil(height / stride_height)

    network = conv_2d(network, 64, 5, activation='relu')  # h = 90, w = 150, d = 64
    network = batch_normalization(network)
    # network = conv_2d(network, 64, 5, activation='relu') # h = 90, w = 150, d = 64
    # network = batch_normalization(network)
    network = max_pool_2d(network, 3, strides=2)  # h = 45, w = 75, d = 64
    network = tf.add(network, side_network)

    network = conv_2d(network, 64, 3, activation='relu')  # h = 45, w = 75, d = 64
    network = batch_normalization(network)
    # network = conv_2d(network, 64, 3, activation='relu') # h = 45, w = 75, d = 64
    # network = batch_normalization(network)
    # network = conv_2d(network, 64, 3, activation='relu') # h = 45, w = 75, d = 64
    # network = batch_normalization(network)
    network = max_pool_2d(network, 3, strides=2)  # h = 23, w = 38, d = 64

    network = conv_2d(network, 64, 3, activation='relu')  # h = 23, w = 38, d = 64
    network = batch_normalization(network)
    network = max_pool_2d(network, 3, strides=2)  # h = 12, w = 19, d = 64

    # network = fully_connected(network, 100, activation='relu')
    # network = dropout(network, 0.5)

    network = fully_connected(network, 50, activation='relu')
    network = dropout(network, 0.5)

    network = fully_connected(network, 5, activation='relu')  # frame -> future area

    # need to write custom loss function, categorical crossentropy is good for binary classification
    # mean_squared_error or mean_pairwise_squared_error
    # think about logistic regression, linear regression
    # logistic regression -> class prediction
    # linear regression -> scalar prediction
    network = regression(network, optimizer='adam', loss=tf.losses.mean_squared_error, learning_rate=1e-3,
                         name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_convnet', max_checkpoints=1, tensorboard_verbose=2,
                        tensorboard_dir='log')

    return model


def train_convnet(model, input, test):
    X = np.array([i[5] for i in input]).reshape(-1, sample_height, sample_width, sample_depth)  # frame
    X1 = [i[0] for i in input]  # previous areas
    X2 = [i[1] for i in input]  # current area
    X3 = [i[3] for i in input]  # mouse position
    X4 = [i[4] for i in input]  # space hit
    XSide = np.concatenate((X1, X2, X3, X4), axis=1)

    Y = [i[2] for i in input]  # future areas

    test_X = np.array([i[5] for i in test]).reshape(-1, sample_height, sample_width, sample_depth)
    test_X1 = [i[0] for i in test]  # previous areas
    test_X2 = [i[1] for i in test]  # current area
    test_X3 = [i[3] for i in test]  # mouse position
    test_X4 = [i[4] for i in test]  # space hit
    test_XSide = np.concatenate((test_X1, test_X2, test_X3, test_X4), axis=1)

    test_Y = [i[2] for i in test]  # future areas

    model.fit([X, XSide], Y, n_epoch=10, validation_set=([test_X, test_XSide], test_Y), snapshot_step=500,
              show_metric=True, run_id='agario-convnet.model')

    filename = 'agario-convnet.model'
    with file_io.FileIO(GS_BUCKET + OUTPUT_FOLDER + filename, "w") as file:
        model.save(file)
    print('saved the model')


def create_model(filename):

    train_data = read_bucket_np(filename)
    test_data_len = len(train_data) // 10
    # print(test_data_len)

    shuffle(train_data)

    input = train_data[:-test_data_len]
    test = train_data[-test_data_len:]
    print('got the data')

    model = convnet()
    print('created the model')

    train_convnet(model, input, test)
    print('trained the model')


if __name__ == '__main__':
    create_model(INPUT_FILENAME)