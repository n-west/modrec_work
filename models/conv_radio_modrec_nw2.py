
from keras import models, layers
from keras.callbacks import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.embeddings import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *


class ConvRadioModrec_exps(object):
    def __init__(self, num_labels):
	taps = 8
	nfilts = 50
        self.model = models.Sequential()
        self.model.add(Convolution2D(nfilts, 1, taps, init='glorot_uniform', border_mode='same', input_shape=(1, 2, 128)))
	self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))

        self.model.add(Convolution2D(nfilts, 1, taps, init='glorot_uniform', border_mode='same'))
	self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(output_dim=num_labels, init="glorot_uniform"))
        self.model.add(Activation("softmax"))

	self.name = "conv-radio-modrec-%ifilts_%itaps_2_layers_1dense" % (nfilts,taps)
	self.desc = "oshea paper, convolutional radio mod rec network"
	self.ver = "v0"


model = ConvRadioModrec_exps
