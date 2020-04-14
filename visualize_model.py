
from keras.models import model_from_yaml
import keras.backend as K
import numpy as np
import seaborn as sns
import data_utils
import matplotlib.pyplot as plt

model_filename = "training_results/cldnn-v4-64.3409"
with open(model_filename + '.yaml', 'r') as model_yamlfile:
    model_yaml = model_yamlfile.read()
model = model_from_yaml(model_yaml)

#with open(model_filename + '.h5') as weights_file:
weights = model.load_weights(model_filename + '.h5')


print model.summary()

#(train_ds, train_labels, valid_ds, valid_labels, snr_valid_ds, snr_valid_labels) = data_utils.what_i_do_in_notebook()

#predicted_classes = model.predict(test_X, batch_size=1024)
#model_accuracy = data_utils.get_model_accuracy(predicted_classes, test_Y)
#print "this model accuracy is %2.3f" % (model_accuracy)
#loss = model.evaluate(test_X, test_Y, batch_size=1024)
#print "this model loss is %2.5f" % loss

#data_utils.make_accuracy_vs_snr(model, snr_valid_ds, snr_valid_labels)
#plt.show()

layer_name = 'convolution2d_2'

def get_layer_from_model(layer_name):
    for layer in model.layers:
        if layer_name == layer.name:
            return layer

def dream_rf():
    input_vec = model.layers[0].input
    filter_index = 0
    
    layer = get_layer_from_model(layer_name)
    input_vec = layer.input
    layer_output = layer.output
    
    print input_vec._keras_shape
    
    for filter_index in range(50):
        # find that layer in our model
        loss = K.mean(layer_output[:, filter_index, :, :])
        
        grads = K.gradients(loss, input_vec)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        
        iterate = K.function([input_vec], [loss, grads])
        
        input_vec_data = np.random.random((1, 80, 2, 128)) * 20 + 128
        for i in range(20):
            loss_value, grads_value = iterate([input_vec_data])
            input_vec_data += grads_value * 1
        
        plt.figure()
        plt.title("filter index %i" % filter_index)
        plt.plot(input_vec_data[0,0,0,:])
        plt.plot(input_vec_data[0,0,1,:])
    plt.show()

#def plot_filters():
layer = get_layer_from_model(layer_name)
print layer.output._keras_shape
weights = model.get_weights()
#for filt in weights[0]:
#    plt.figure()
#    plt.subplot(211)
#    plt.title("filter taps")
#    plt.plot(filt[0,0,:])
#    plt.subplot(212)
#    plt.title("filter freq response mag")
#    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(filt[0,0,:], n=128))))

plt.show()
