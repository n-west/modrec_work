import numpy as np
import pickle
import matplotlib.pyplot as plt

from keras import models, layers
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.embeddings import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

def calc_vec_energy(vec):
    isquared = np.power(vec[0],2.0)
    qsquared = np.power(vec[1], 2.0)
    inst_energy = np.sqrt(isquared+qsquared)
    return sum(inst_energy)

def calc_mod_energies(ds):
    for modulation, snr in ds:
        if snr == 18:
            avg_energy = 0
            for vec in ds[(modulation, snr)]:
                avg_energy += calc_vec_energy(vec)
            avg_energy /= ds[(modulation, snr)].__len__()
            print "%s at %i has avg energy of %2.1f" % (modulation, snr, avg_energy)

def plot_sanity_vec(vec, mod_type):
    plt.figure()
    plt.title(mod_type)
    plt.plot(vec[0])
    plt.plot(vec[1])
    plt.draw()

def open_ds(location="/home/nathan/datasets/RML2016.10a_dict.dat"):
    f = open(location)
    ds = pickle.load(f)
    return ds


### The following come from my ipython notebook
def reformat_ds(ds, labels_onehot_dict, modulation):
    """
    Transform dataset from an array of [i],[q] arrays to an array of [i,q]
    :param ds:
    :return:
    """
    num_vecs, nchans, vec_length = ds.shape
    transformed_ds = np.zeros((num_vecs, nchans, vec_length), dtype=ds.dtype)
    ds_labels = np.zeros((num_vecs, num_labels))
    indx = 0
    for vec in ds:
        #transformed_ds[indx][0] = vec[0]
        #transformed_ds[indx][1] = vec[1]
        ds_labels[indx] = labels_onehot_dict[modulation]
        indx += 1
    return (ds, ds_labels)

def onehot_to_class(onehot_labels):
    """
    A utility for converting onehot encodings to a class number
    :param onehot_labels: a numpy array with a onehot encodings
    :return: a numpy array with classes
    """
    return np.array([np.argmax(onehot) for onehot in onehot_labels])

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def extract_desired_snrs(ds, labels_onehot_dict, desired_snr=None):
    nvecs = 0
    for modulation, snr in ds.keys():
        if desired_snr == snr or desired_snr is None:
            nvecs += ds[(modulation, desired_snr)].shape[0]

    ds_vecs = np.empty((nvecs, 2, 128), dtype=np.float32)
    ds_labels = np.empty((nvecs, num_labels), dtype=np.float32)

    # extract and reformat the data we care about
    indx = 0
    for modulation, snr in ds.keys():
        if desired_snr == snr or desired_snr == None:
            # This really just gives us a one-hot encoded vector
            (mod_vecs, mod_labels) = reformat_ds(ds[(modulation, snr)], labels_onehot_dict, modulation)
            nmodvecs = mod_vecs.shape[0]
            ds_vecs[indx:indx+nmodvecs] = mod_vecs[:]
            ds_labels[indx:indx+nmodvecs] = mod_labels[:]
            indx += nmodvecs
    # scramble the data
    vec_size = ds_vecs.shape[1]
    permuter = np.random.permutation(nvecs)
    shuffled_ds = ds_vecs[permuter]
    shuffled_labels = ds_labels[permuter]
    return (shuffled_ds, shuffled_labels, nvecs)

num_labels = 0
labels_onehot = []
labels_onehot_dict = {}
def what_i_do_in_notebook():
    global num_labels # gross
    global labels_onehot
    global labels_onehot_dict
    ds = open_ds()
    label_list = list(set([x[0] for x in ds.keys()]))

    snrs = []
    total_nvecs = 0
    for modulation, snr in ds.keys():
        snrs.append(snr)
        total_nvecs += ds[(modulation, snr)].shape[0]
    snrs.sort()
    snrs = set(snrs)

    # get all of the mods
    label_list = list(set([x[0] for x in ds.keys()]))
    # Make the one-hot label arrays
    num_labels = label_list.__len__()
    for mod_indx in range(num_labels):
        labels_onehot.append(np.zeros([num_labels]))
        labels_onehot[mod_indx][mod_indx] = 1
        labels_onehot_dict[label_list[mod_indx]] = labels_onehot[mod_indx]

    snr_keyed_ds = {}
    snr_keyed_labels = {}
    snr_keyed_validation_data = {}
    snr_keyed_validation_labels = {}

    train_size = int(total_nvecs * .8)
    per_snr_train_size = train_size / 20
    valid_size = total_nvecs - 20*per_snr_train_size
    per_snr_valid_size = valid_size/20
    training_data = np.zeros((20*per_snr_train_size, 2, 128))
    training_labels = np.zeros((20*per_snr_train_size, label_list.__len__()))
    valid_data = np.zeros((20*per_snr_valid_size, 2, 128))
    valid_labels = np.zeros((20*per_snr_valid_size, label_list.__len__()))
    train_indx = 0
    valid_indx = 0
    snrs_list = list(snrs)
    snrs_list.sort()
    this_total_nvecs = 0
    for snr in snrs_list:
        (this_snr_ds, this_snr_labels, nvecs) = extract_desired_snrs(ds, labels_onehot_dict, snr)
        snr_keyed_ds[snr] = this_snr_ds
        snr_keyed_labels[snr] = this_snr_labels
        training_data[train_indx:train_indx+per_snr_train_size][:][:] = snr_keyed_ds[snr][0:per_snr_train_size]
        training_labels[train_indx:train_indx+per_snr_train_size][:] = snr_keyed_labels[snr][0:per_snr_train_size]

        valid_data[valid_indx:valid_indx+per_snr_valid_size][:][:] = snr_keyed_ds[snr][per_snr_train_size:per_snr_train_size+per_snr_valid_size]
        valid_labels[valid_indx:valid_indx+per_snr_valid_size][:] = snr_keyed_labels[snr][per_snr_train_size:per_snr_train_size+per_snr_valid_size]

        snr_keyed_validation_data[snr] = valid_data[valid_indx:valid_indx+per_snr_valid_size][:][:]
        snr_keyed_validation_labels[snr] = valid_labels[valid_indx:valid_indx+per_snr_valid_size][:]
        train_indx += per_snr_train_size
        valid_indx += per_snr_valid_size

    train_permuter = np.random.permutation(training_data.shape[0])
    train_ds = training_data[train_permuter].reshape(training_data.shape[0], 1, 2, 128)
    train_labels = training_labels[train_permuter]
    valid_ds = valid_data.reshape(valid_data.shape[0], 1, 2, 128)
    return (train_ds, train_labels, valid_ds, valid_labels, snr_keyed_validation_data, snr_keyed_validation_labels)


def get_model_accuracy(predicted_labels, valid_labels):
    # Convert classes to onehot encoding
    #classes_onehot = np.zeros([valid_labels.__len__(), num_labels])
    #indx = 0
    ## Get a one-hot classification from class number
    #for predicted_label in predicted_labels:
    #    classes_onehot[indx] = labels_onehot[predicted_label]
    #    indx += 1
    return accuracy(predicted_labels, valid_labels)

def plot_confusion_matrix(cm, target_names=None, title='Confusion matrix', cmap=plt.cm.Oranges):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def make_confusion(classes, valid_labels):
    # onehot_labels_dict has keys that are string labels and values that are one-hot encodings
    # This is probably in order, but we should make a new list of labels that is in order of class
    # First make a dict of class: 'string of class'
    inorder_classes = dict()
    for k in labels_onehot_dict:
        inorder_classes[np.argmax(labels_onehot_dict[k])] = k

    inorder_labels = tuple([inorder_classes[c] for c in xrange(inorder_classes.__len__())])

    cm = confusion_matrix(classes, onehot_to_class(valid_labels))
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure()
    plot_confusion_matrix(cm_normalized, target_names=inorder_labels, title="Radio CMFDNN Confusion")
    fig.set_facecolor("white")
    #plt.savefig("CLDNN matrix:50epoch,ConvMaxLstmDenseDense.png")
    return fig

def make_accuracy_vs_snr(classifier_model, snr_keyed_validation_data, snr_keyed_validation_labels):
    num_snrs = snr_keyed_validation_data.keys().__len__()
    model_accuracy = np.zeros(num_snrs)
    snrs_array = np.zeros(num_snrs)
    snr_indx = 0
    for (snr, vectors) in snr_keyed_validation_data.items():
        predictions = classifier_model.predict(vectors.reshape(vectors.shape[0],1,2,128))
        #classes_onehot = np.zeros([predictions.shape[0], num_labels])
        #indx = 0
        #for clasify in predictions:
        #    classes_onehot[indx] = labels_onehot[clasify]
        #    indx += 1
        this_accuracy = accuracy(predictions, snr_keyed_validation_labels[snr])
        print "%i dB is at %2.3f" % (snr,this_accuracy)
        model_accuracy[snr_indx] = this_accuracy
        snrs_array[snr_indx] = snr
        snr_indx += 1
    fig = plt.figure()
    plt.scatter(snrs_array, model_accuracy)
    plt.title("Model Accuracy vs SNR")
    return fig

def main():
    ds = open_ds()
    plot_sanity_vec(ds[("BPSK", 18)][50], "BPSK")
    plot_sanity_vec(ds[("QAM64", 18)][50], "QAM64")
    plot_sanity_vec(ds[("AM-SSB", 18)][50], "AM-SSB")
    calc_mod_energies(ds)
    plt.show()

if __name__ == "__main__":
    main()
