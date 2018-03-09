from time import time
import argparse
from io import BytesIO

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import  TensorBoard, EarlyStopping
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Reshape, BatchNormalization, concatenate, Flatten, \
    Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.lib.io import file_io

tf.reset_default_graph()

SAMPLE_HEIGHT = 90
SAMPLE_WIDTH = 150
SAMPLE_DEPTH = 1
input_shape = (SAMPLE_HEIGHT, SAMPLE_WIDTH, SAMPLE_DEPTH)
LEARN_RATE = 1.0e-4
NUM_EPOCH = 3


def createGenerator(X, I, Y):
    while True:
        # suffled indices
        idx = np.random.permutation( X.shape[0])
        # create image generator
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=10, #180,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1, #0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1, #0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False)  # randomly flip images

        batches = datagen.flow( X[idx], Y[idx], batch_size=64, shuffle=False)
        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]

            yield [batch[0], I[ idx[ idx0:idx1 ] ]], batch[1]

            idx0 = idx1
            if idx1 >= X.shape[0]:
                break

def load_data(train_file):
    f = BytesIO(file_io.read_file_to_string(train_file, binary_mode=True))
    train_data = np.load(f)
    test_data_len = len(train_data) // 10
    input = train_data
    test = train_data[-test_data_len:]

    X = np.array([i[5] for i in input]).reshape(-1, SAMPLE_HEIGHT, SAMPLE_WIDTH, SAMPLE_DEPTH)  # frame
    X1 = [i[0] for i in input]  # previous areas
    X2 = [i[1] for i in input]  # current area
    X3 = [i[3] for i in input]  # mouse position
    X4 = [i[4] for i in input]  # space hit
    XSide = np.concatenate((X1, X2, X3, X4), axis=1)

    Y = np.array([i[2] for i in input])  # future areas

    test_X = np.array([i[5] for i in test]).reshape(-1, SAMPLE_HEIGHT, SAMPLE_WIDTH, SAMPLE_DEPTH)
    test_X1 = [i[0] for i in test]  # previous areas
    test_X2 = [i[1] for i in test]  # current area
    test_X3 = [i[3] for i in test]  # mouse position
    test_X4 = [i[4] for i in test]  # space hit
    test_XSide = np.concatenate((test_X1, test_X2, test_X3, test_X4), axis=1)

    test_Y = np.array([i[2] for i in test])  # future areas

    return X,XSide,Y,test_X,test_XSide,test_Y


def cnn_model():
    # main input
    main_input = Input(shape=input_shape, name='main_input')
    print(main_input)
    main = Conv2D(64, 5, padding='same', activation='relu')(main_input)

    main = BatchNormalization()(main)
    main = MaxPooling2D(3, strides=2, padding='same')(main)

    # extra_input
    extra_input = Input(shape=(16,), name='extra_input')
    extra = Dense(64, activation='relu')(extra_input)
    extra = Lambda(lambda x: K.tile(x, [1, 3375]))(extra)
    extra = Reshape([45, 75, 64])(extra)

    combined = concatenate([main, extra])

    combined = Conv2D(64, 3, padding='same', activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = MaxPooling2D(3, strides=2, padding='same')(combined)

    combined = Conv2D(32, 3, padding='same', activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = MaxPooling2D(3, strides=2, padding='same')(combined)

    combined = Flatten()(combined)
    combined = Dense(32, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    main_output = Dense(5, activation='sigmoid', name='main_output')(combined)

    model = Model(inputs=[main_input,extra_input], outputs=main_output)
    model.summary()

    return model


def train_model(train_file, job_dir, augmented, **args):
    print("Use this line to access tensorboard")
    print("tensorboard --logdir="+job_dir+"logs --port 8000 --reload_interval=5")
    X, XSide, Y, test_X, test_XSide, test_Y = load_data(train_file)
    model = cnn_model()

    # if want to use add checkpoint to callbacks in fit, currently not avaliable on gcloud
    # if checkpoint used do not need model.save
    #checkpoint = ModelCheckpoint('best_model_improved.h5',  # model filename
    #                             monitor='val_loss',  # quantity to monitor
    #                             verbose=0,  # verbosity - 0 or 1
    #                             save_best_only=True,  # The latest best model will not be overwritten
    #                             mode='auto')  # The decision to overwrite model is made automatically depending on the quantity to monitor

    tensorboard = TensorBoard(log_dir=job_dir+'logs')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    model.compile(loss='categorical_crossentropy',  # Better loss function for neural networks
                  optimizer=Adam(lr=LEARN_RATE),  # Adam optimizer with 1.0e-4 learning rate
                  metrics=['accuracy'])  # Metrics to be evaluated by the model


    if augmented == "False":
        model_details = model.fit([X,XSide], Y,
                                  batch_size=64,
                                  epochs=NUM_EPOCH,  # number of iterations
                                  validation_split=0.2,
                                  shuffle=True,
                                  callbacks=[tensorboard,early_stopping],
                                  verbose=1)
    else:
        datagen = createGenerator(X,XSide,Y)

        model_details = model.fit_generator(datagen,steps_per_epoch=len(X) / 64, # number of samples per gradient update
                                                    epochs=NUM_EPOCH,  # number of iterations
                                                    validation_data=([test_X,test_XSide], test_Y),
                                                    callbacks=[tensorboard,early_stopping],
                                                    verbose=1)

    score = model.evaluate([test_X, test_XSide], test_Y, batch_size=64)
    print("Accuracy: {0:.2f}".format(score[1]*100))

    model.save('model.h5')
    # Save model.h5 on to google storage
    if str(job_dir).startswith("gs://"):
        with file_io.FileIO('model.h5', mode='rb') as input_f:
            with file_io.FileIO(job_dir + 'model.h5', mode='w+') as output_f:
                output_f.write(input_f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-file',
        help='GCS or local paths to training data',
        required=True
    )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    parser.add_argument(
        '--augmented',
        help='True for augmented data, False for non-augmented data',
        required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__

    train_model(**arguments)