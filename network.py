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
    network = input_data(shape=[None, config.sample_width, config.sample_height, config.sample_depth], name='input')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, config.actions, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=1e-3, name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

    return model

def train_alexnet(model, input, test, config):
    X = np.array([i[5] for i in input]).reshape(-1, config.sample_width, config.sample_height, config.sample_depth)
    Y = [i[3] for i in input]

    test_x = np.array([i[5] for i in test]).reshape(-1, config.sample_width, config.sample_height, config.sample_depth)
    test_y = [i[3] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id='agario-alexnet.model')
    
    model.save('agario-alexnet.model')

def customized_loss(y_pred, y_true, loss='euclidian'):
    # euclidian distance loss
    if loss=='euclidian':
        val = tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))
    return val

def CNN(config):
    previous_area_input_layer = input_data(shape=[None, 5, 1], name='prev')
    current_area_input_layer = input_data(shape=[None, 1], name='current')
    mouse_input_layer = input_data(shape=[None, 9, 1], name='mouse')
    space_input_layer = input_data(shape=[None, 1], name='space')
    frame_input_layer = input_data(shape=[None, config.sample_width, config.sample_height, config.sample_depth], name='frame')

    first_convolutional_layer = conv_2d(frame_input_layer, 96, 11, strides=4, activation='relu')
    first_pooling_layer = max_pool_2d(first_convolutional_layer, 3, strides=3)
    first_normalization_layer = local_response_normalization(first_pooling_layer)

    second_convolutional_layer = conv_2d(first_normalization_layer, 256, 11, strides=4, activation='relu')
    second_pooling_layer = max_pool_2d(second_convolutional_layer, 3, strides=3)
    second_normalization_layer = local_response_normalization(second_pooling_layer)

    third_convolutional_layer = conv_2d(second_normalization_layer, 256, 11, strides=4, activation='relu')
    third_pooling_layer = max_pool_2d(third_convolutional_layer, 3, strides=3)
    third_normalization_layer = local_response_normalization(third_pooling_layer)

    '''first_part_output = tflearn.flatten(third_convolutional_layer)

    merged_output = tf.concat([previous_area_input_layer, current_area_input_layer, mouse_input_layer, space_input_layer, first_part_output], 0)

    first_dense_layer = fully_connected(merged_output, 1024, activation='relu')
    first_dropout_layer = dropout(first_dense_layer, .5)
        
    second_dense_layer = fully_connected(first_dropout_layer, 100, activation='relu')
    second_dropout_layer = dropout(second_dense_layer, .5)

    third_dense_layer = fully_connected(second_dropout_layer, 50, activation='relu')
    third_dropout_layer = dropout(third_dense_layer, .5)

    final_layer = fully_connected(third_dropout_layer, config.actions, activation='softmax')'''

    first_dense_layer = fully_connected(third_normalization_layer, 1024, activation='relu')
    first_dropout_layer = dropout(first_dense_layer, .5)

    second_dense_layer = fully_connected(first_dropout_layer, 100, activation='relu')
    second_dropout_layer = dropout(second_dense_layer, .5)

    third_dense_layer = fully_connected(second_dropout_layer, 50, activation='relu')
    third_dropout_layer = dropout(third_dense_layer, .5)

    final_layer_first_part = fully_connected(third_dropout_layer, config.actions, activation='softmax')
    final_layer_previous_area_part = fully_connected(previous_area_input_layer, config.actions, activation='softmax')
    final_layer_current_area_part = fully_connected(current_area_input_layer, config.actions, activation='softmax')
    final_layer_mouse_part = fully_connected(mouse_input_layer, config.actions, activation='softmax')
    final_layer_space_part = fully_connected(space_input_layer, config.actions, activation='softmax')
    
    final_layer = tf.concat([final_layer_previous_area_part, final_layer_current_area_part, final_layer_mouse_part, final_layer_space_part, final_layer_first_part], 0)

    # need to write our own loss function
    regression_layer = regression(final_layer, optimizer='momentum', loss=customized_loss, learning_rate=1e-3, name='target')

    model = tflearn.DNN(regression_layer, checkpoint_path='model', max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
    return model

def train_model(model, input, test, config):
    # need to think about sizes of the every layer
    prev = np.array([i[0] for i in input]).reshape(-1, 5, 1, 1)
    current = np.array([i[1] for i in input]).reshape(-1, 1, 1, 1)
    future = np.array([i[2] for i in input]).reshape(-1, 5, 1, 1) # output
    mouse = np.array([i[3] for i in input]).reshape(-1, 9, 1, 1)
    space = np.array([i[4] for i in input]).reshape(-1, 1, 1, 1)
    frame = np.array([i[5] for i in input]).reshape(-1, config.sample_width, config.sample_height, config.sample_depth)

    test_prev = np.array([i[0] for i in test]).reshape(-1, 5, 1, 1)
    test_current = np.array([i[1] for i in test]).reshape(-1, 1, 1, 1)
    test_future = np.array([i[2] for i in test]).reshape(-1, 5, 1, 1) # output
    test_mouse = np.array([i[3] for i in test]).reshape(-1, 9, 1, 1)
    test_space = np.array([i[4] for i in test]).reshape(-1, 1, 1, 1)
    test_frame = np.array([i[5] for i in test]).reshape(-1, config.sample_width, config.sample_height, config.sample_depth)
    
    model.fit({'prev': prev, 'current': current, 'mouse': mouse, 'space': space, 'frame': frame}, {'target': future},
              n_epoch=8,
              validation_set=({'prev': test_prev, 'current': test_current, 'mouse': test_mouse, 'space': test_space, 'frame': test_frame}, {'target': test_future}),
              snapshot_step=250,
              show_metric=True,
              run_id='agario-v0.model')
    model.save('agario-v0.model')

if __name__ == '__main__':
    config = Config()

    train_data = np.load('training_data.npy')
    test_data_len = len(train_data) // 10
    print(test_data_len)
    input = train_data[:-test_data_len]
    test = train_data[-test_data_len:]

    #model = CNN(config)
    #train_model(model, input, test, config)

    model = alexnet(config)
    train_alexnet(model, input, test, config)