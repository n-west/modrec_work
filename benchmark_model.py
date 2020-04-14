
import data_utils

from keras import models, layers
from keras.optimizers import *
from keras.callbacks import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.embeddings import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *

import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time
import datetime

today = datetime.datetime.now().ctime()

model_to_test = 'conv_radio_modrec' if len(sys.argv)<2 else sys.argv[1]
print model_to_test

(train_ds, train_labels, valid_ds, valid_labels, snr_valid_ds, snr_valid_labels) = data_utils.what_i_do_in_notebook()

# dynamically load a model. default to osh, but accept cmd line arg
import models
model_container = eval('models.' + model_to_test)
mut = model_container.model(data_utils.num_labels)
model = mut.model
name = mut.name
desc = mut.desc
ver = mut.ver

# train that model
schedule = np.append(np.ones(15,dtype='float32')*.1, np.ones(15)*.05)
schedule = np.append(schedule, np.ones(70)*.01)
schedule = np.append(schedule, np.ones(400)*.001)
def sched(epoch):
	return float(schedule[epoch])
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam')
start_time = time.clock()
history = model.fit(train_ds, train_labels, nb_epoch=500, batch_size=1024,
                shuffle=True, validation_split=.1,
		callbacks=[BaseLogger(),
		ModelCheckpoint('training_results/%s-weights.{epoch:02d}-{val_loss:.2f}.hdf5'%name, monitor='val_loss', save_best_only=True, mode='min'),
			EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min'),
		])
stop_time = time.clock()

training_time = stop_time - start_time

# test performance
start_time = time.clock()
predicted_classes = model.predict(valid_ds, batch_size=1024)
stop_time = time.clock()
model_accuracy = data_utils.get_model_accuracy(predicted_classes, valid_labels)
print "this model accuracy is %2.3f" % (model_accuracy)
loss = model.evaluate(valid_ds, valid_labels, batch_size=1024)
print "this model loss is %2.5f" % loss

prediction_time = stop_time - start_time

model_yaml = model.to_yaml()
f = open("training_results/%s-%s-%2.4f.yaml"%(name, ver, model_accuracy), "w")
f.write(model_yaml)
f.close()
model.save_weights("training_results/%s-%s-%2.4f.h5"%(name, ver, model_accuracy), overwrite=True)


#confusion_fig = data_utils.make_confusion(predicted_classes, valid_labels)
accuracy_fig = data_utils.make_accuracy_vs_snr(model, snr_valid_ds, snr_valid_labels)
plt.figure()
plt.title('Training Performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')

train_log = open('training_log.json', 'a')
train_log.write("{\n")
train_log.write("  'date': %s\n" % today)
train_log.write("  'model': %s - %s\n" %(name,ver))
train_log.write("  'model desc': %s\n" % desc)
train_log.write("  'train time': %s\n" % training_time)
train_log.write("  'prediction time': %s\n" % prediction_time)
train_log.write("  'class_accuracy': %2.3f\n" % model_accuracy)
train_log.write("  'train_loss': %2.5f\n" % loss)
train_log.write("}\n\n")

plt.show()

