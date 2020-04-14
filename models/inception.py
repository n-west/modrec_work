
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
	b1 = Convolution2D(50, 1, 1, init='glorot_uniform', border_mode='same', activation="relu")(inputs)
	b1 = Convolution2D(50, 2, 8, init='glorot_uniform', border_mode='same', activation="relu")(b1)

	b2 = Convolution2D(50, 1, 1, init='glorot_uniform', border_mode='same', activation="relu")(inputs)
	b2 = Convolution2D(50, 2, 3, init='glorot_uniform', border_mode='same', activation="relu")(b2)

	b3 = Convolution2D(50, 1, 1, init='glorot_uniform', border_mode='same', activation="relu")(inputs)
	top = merge([b1, b2, b3], mode='concat', concat_axis=1)
	top = Dropout(0.6)(top)

	b1 = Convolution2D(50, 1, 1, init='glorot_uniform', border_mode='same', activation="relu")(top)
	b1 = Convolution2D(50, 2, 8, init='glorot_uniform', border_mode='same', activation="relu")(b1)

	b2 = Convolution2D(50, 1, 1, init='glorot_uniform', border_mode='same', activation="relu")(top)
	b2 = Convolution2D(50, 2, 3, init='glorot_uniform', border_mode='same', activation="relu")(b2)

	b3 = Convolution2D(50, 1, 1, init='glorot_uniform', border_mode='same', activation="relu")(top)
	top = merge([b1, b2, b3], mode='concat', concat_axis=1)
	top = Dropout(0.6)(top)

	b1 = Convolution2D(50, 1, 1, init='glorot_uniform', border_mode='same', activation="relu")(top)
	b1 = Convolution2D(50, 2, 8, init='glorot_uniform', border_mode='same', activation="relu")(b1)

	b2 = Convolution2D(50, 1, 1, init='glorot_uniform', border_mode='same', activation="relu")(top)
	b2 = Convolution2D(50, 2, 3, init='glorot_uniform', border_mode='same', activation="relu")(b2)

	b3 = Convolution2D(50, 1, 1, init='glorot_uniform', border_mode='same', activation="relu")(top)
	top = merge([b1, b2, b3], mode='concat', concat_axis=1)
	top = Dropout(0.6)(top)

	top = Flatten()(top)
	top = Dense(256, activation="relu")(top)
	predictions = Dense(output_dim=num_labels, init="glorot_uniform", activation="softmax")(top)
	
	self.model = models.Model(input=inputs, output=predictions)

	self.name = "inception3do"
	self.desc = "3 inception modules with do"
	self.ver = "v0"


model = resnet
