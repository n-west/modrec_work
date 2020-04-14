
from keras import models
from keras.callbacks import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.embeddings import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *


class rescldnn(object):
    def __init__(self, num_labels):

	inputs = Input(shape=(1,2,128))
	l1 = Convolution2D(64, 1, 8, init='glorot_uniform', border_mode='same', activation="relu")(inputs)
	do1 = Dropout(0.6)(l1)
	l2 = Convolution2D(64, 1, 8, init='glorot_uniform', border_mode='same', activation="relu")(do1)
	do2 = Dropout(0.6)(l2)
	res1 = merge([l1,l2], mode='sum')

	l3 = Convolution2D(64, 1, 8, init='glorot_uniform', border_mode='same', activation="relu")(res1)
	do3 = Dropout(0.6)(l3)
	l4 = Convolution2D(64, 1, 8, init='glorot_uniform', border_mode='same', activation="relu")(do3)
	do4 = Dropout(0.6)(l4)
	res2 = merge([res1,l4], mode='sum')


	top = TimeDistributed(LSTM(40))(res2)
	do4 = Dropout(0.6)(top)

	top = Flatten()(do4)
	top = Dense(256, activation="relu")(top)
	top = Dropout(0.6)(top)
	top = Dense(128, activation="relu")(top)
	top = Dropout(0.6)(top)
	predictions = Dense(output_dim=num_labels, init="glorot_uniform", activation="softmax")(top)
	
	self.model = models.Model(input=inputs, output=predictions)

	self.name = "resnet-cldnn"
	self.desc = "4 64 1x8 convolutions with 2 residual, lstm, 2 smaller dense, 0.6 dropout"
	self.ver = "v0"


model = rescldnn
