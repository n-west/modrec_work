
from keras import models
from keras.callbacks import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.embeddings import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *


class resnet(object):
    def __init__(self, num_labels):

	inputs = Input(shape=(1,2,128))
	l1 = Convolution2D(64, 1, 8, init='glorot_uniform', border_mode='same', activation="relu")(inputs)
	l2 = Convolution2D(64, 1, 8, init='glorot_uniform', border_mode='same', activation="relu")(l1)
	l3 = Convolution2D(64, 1, 8, init='glorot_uniform', border_mode='same', activation="relu")(l2)
	res1 = merge([l1,l3], mode='sum')

	l4 = Convolution2D(64, 1, 8, init='glorot_uniform', border_mode='same', activation="relu")(res1)
	l5 = Convolution2D(64, 1, 8, init='glorot_uniform', border_mode='same', activation="relu")(l4)
	res2 = merge([res1,l5], mode='sum')
	l6 = Convolution2D(64, 1, 8, init='glorot_uniform', border_mode='same', activation="relu")(res2)
	do4 = Dropout(0.6)(l6)

	top = Flatten()(do4)
	top = Dense(256, activation="relu")(top)
	top = Dropout(0.6)(top)
	top = Dense(128, activation="relu")(top)
	top = Dropout(0.6)(top)
	predictions = Dense(output_dim=num_labels, init="glorot_uniform", activation="softmax")(top)
	
	self.model = models.Model(input=inputs, output=predictions)

	self.name = "functional net test"
	self.desc = "4 64 1x8 convolutions with 2 residual, 2 smaller dense, 0.6 dropout"
	self.ver = "v0"


model = resnet
