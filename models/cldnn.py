
from keras import models, layers
from keras.callbacks import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.embeddings import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *


class cldnn(object):
    def __init__(self, num_labels):
        self.model = models.Sequential()
        self.model.add(Convolution2D(80, 1, 8, init='glorot_uniform', border_mode='same', input_shape=(1, 2, 128)))
	self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Convolution2D(80, 1, 8, init='glorot_uniform', border_mode='same'))
	self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Convolution2D(80, 1, 8, init='glorot_uniform', border_mode='same'))
	self.model.add(Activation("relu"))
        self.model.add(TimeDistributed(LSTM(64)))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256))
        self.model.add(Dropout(0.5))
        self.model.add(Activation("relu"))
        self.model.add(Dense(output_dim=num_labels, init="glorot_uniform"))
        self.model.add(Activation("softmax"))

	self.name = "cldnn"
	self.desc = "3 smallish convolutions, lstm, 2 dense"
	self.ver = "v4"


model = cldnn
