import tflearn

import tensorflow as tf

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

def CNN(input, config):
    previous_area_input = input[0]
    current_area_input = input[1]
    future_area_input = input[2]
    mouse_input = input[3]
    space_input = input[4]
    image_input = input[5]

    input_layer = input_data(shape=[None, config.sample_width, config.sample_height, config.samle_depth], name='main-input')
    first_convolutional_layer = conv_2d(input_layer, 96, 11, strides=4, activation='relu')
    first_pooling_layer = max_pool_2d(first_convolutional_layer, 3, strides=3)
    first_normalization_layer = local_response_normalization(first_pooling_layer)

    second_convolutional_layer = conv_2d(first_normalization_layer, 256, 11, strides=4, activation='relu')
    second_pooling_layer = max_pool_2d(second_convolutional_layer, 3, strides=3)
    secon_normalization_layer = local_response_normalization(second_pooling_layer)

    third_convolutional_layer = conv_2d(secon_normalization_layer, 256, 11, strides=4, activation='relu')
    third_pooling_layer = max_pool_2d(third_convolutional_layer, 3, strides=3)
    thid_normalization_layer = local_response_normalization(third_pooling_layer)

    first_part_output = tflearn.flatten(third_convolutional_layer)
    merged_output = tf.concat([previous_area_input, current_area_input, future_area_input, mouse_input, space_input, first_part_output], 0) #future area will be output

    first_dense_layer = fully_connected(merged_output, 1024, activation='relu')
    first_dropout_layer = dropout(first_dense_layer, .5)
        
    second_dense_layer = fully_connected(first_dropout_layer, 100, activation='relu')
    second_dropout_layer = dropout(second_dense_layer, .5)

    third_dense_layer = fully_connected(second_dropout_layer, 50, activation='relu')
    third_dropout_layer = dropout(third_dense_layer, .5)

    final_layer = fully_connected(third_dropout_layer, config.actions, activation='softmax')

    return final_layer