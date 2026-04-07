import keras
import simon
import numpy as np
from pickle import dump
import os
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, Add
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model

from tensorflow.keras.utils import plot_model

from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制大部分 TensorFlow 的日志消息。这可以使您在使用 TensorFlow 时减少控制台输出的冗长性
wdir = './SENet_result/'



def make_resnet(num_blocks=4, num_filters=5, num_outputs=2, d1=30, d2=30, INPUT_SIZE=8, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid'):
    #Input layer
    inp = Input(shape=(num_blocks * INPUT_SIZE, ));
    rs = Reshape((num_blocks, INPUT_SIZE))(inp);
    perm = Permute((2,1))(rs);

    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
    conv0 = BatchNormalization()(conv0);
    conv0 = Activation('relu')(conv0);
    #iteration residual layers
    shortcut = conv0;
    for i in range(depth):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
        conv1 = BatchNormalization()(conv1);
        conv1 = Activation('relu')(conv1);
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1);
        conv2 = BatchNormalization()(conv2);
        conv2 = Activation('relu')(conv2);
        shortcut = Add()([shortcut, conv2]);
    #prediction layers
    flat1 = Flatten()(shortcut);
    dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1);
    dense1 = BatchNormalization()(dense1);
    dense1 = Activation('relu')(dense1);
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
    dense2 = BatchNormalization()(dense2);
    dense2 = Activation('relu')(dense2);
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2);
    model = Model(inputs=inp, outputs=out);
    model.summary()
    plot_model(model, to_file="LinearNet_model.png", show_shapes=True, show_dtype=False, show_layer_names=True,
               rankdir="TB", expand_nested=True, dpi=96)
    return(model);


bs=200
num_epochs = 200
num_rounds=3
depth=2
size_train = 10 ** 6
size_eval = 10 ** 5
pad =8
label_num=4
mark = []
wdir = '../model/exdes1/freshly_trained_nets/'
dataset = "../data/exdes1/dataset_liner/"

make_resnet(INPUT_SIZE=pad, num_filters = pad, num_outputs = label_num, d1 = 4*pad, d2 = 4*pad, depth = depth, final_activation = 'softmax')