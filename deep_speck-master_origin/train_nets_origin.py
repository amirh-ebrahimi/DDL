import speck_origin as sp

import numpy as np

import pickle

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2

bs = 5000
wdir = '/content/gdrive/My Drive/Colab Notebooks/'


def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)
    return (res)


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return res


def make_net(num_input=32, num_outputs=1, d_arr=None, reg_param=0.0001, final_activation='sigmoid'):
    if d_arr is None:
        d_arr = [32, 64, 32]
    inp = Input(shape=(num_input,))
    shortcut = inp
    for d in d_arr:
        dense = Dense(d, kernel_regularizer=l2(reg_param))(shortcut)
        dense = BatchNormalization()(dense)
        dense = Activation('relu')(dense)
        shortcut = dense
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense)
    model = Model(inputs=inp, outputs=out)
    return model


def train_speck_distinguisher1(num_epochs, i, j, num_rounds=7, d_arr=None):
    # create the network
    if d_arr is None:
        d_arr = [32, 64, 32]
    net = make_net(num_input=16, d_arr=d_arr, reg_param=10 ** -5)
    depth = len(d_arr)
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    # generate training and validation data
    X, Y = sp.make_train_data(10 ** 7, num_rounds, i)
    X_eval, Y_eval = sp.make_train_data(10 ** 6, num_rounds, i)
    # set up model checkpoint
    check = make_checkpoint(wdir + 'best' + str(num_rounds) + 'trun' + str(j) + 'depth' + str(depth) + '.h5')
    # create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    # train and evaluate
    h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr, check])
    np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '.npy', h.history['val_acc'])
    np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '.npy', h.history['val_loss'])
    dump(h.history, open(wdir + 'hist' + str(num_rounds) + 'r_depth' + str(depth) + '.p', 'wb'))
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    return net, h


# make residual tower of convolutional blocks
def make_resnet(num_blocks=2, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3, depth=5,
                reg_param=0.0001, final_activation='sigmoid'):
    # Input and preprocessing layers
    inp = Input(shape=(num_blocks * word_size,))
    rs = Reshape((num_blocks, word_size))(inp)
    perm = Permute((2, 1))(rs)
    # add a single residual layer that will expand the data to num_filters channels
    # this is a bit-sliced layer
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    # add residual blocks
    shortcut = conv0
    for i in range(depth):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
    # add prediction head
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return model


def train_speck_distinguisher2(num_epochs, num_rounds=7, depth=2):
    # create the network
    net = make_resnet(depth=depth, reg_param=10 ** -5)
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    # generate training and validation data
    X, Y = sp.make_train_data(10 ** 7, num_rounds)
    X_eval, Y_eval = sp.make_train_data(10 ** 6, num_rounds)
    # set up model checkpoint
    check = make_checkpoint(wdir + 'best' + str(num_rounds) + 'depth' + str(depth) + '.h5')
    # create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    # train and evaluate
    h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr, check])
    np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '.npy', h.history['val_acc'])
    np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '.npy', h.history['val_loss'])
    dump(h.history, open(wdir + 'hist' + str(num_rounds) + 'r_depth' + str(depth) + '.p', 'wb'))
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    return net, h
