
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle

colors = ["red", "green", "blue", "yellow", "purple", "black", "orange", "grey", "lawngreen", "cyan", "khaki"]

# Plot the number of filters / feature maps
(res_epochs, res_history) = cPickle.load(open("deep_resnet_history.pickle", "rb"))
(cnn_epochs, cnn_history) = cPickle.load(open("cnn_history.pickle", "rb"))

plt.plot(res_epochs, res_history['loss'], color="blue", label="Resnet Training Loss")
plt.plot(res_epochs, res_history['val_loss'], color="cyan", label="Resnet Validation Loss")

plt.plot(cnn_epochs, cnn_history['loss'], color="red", label="CNN Training Loss")
plt.plot(cnn_epochs, cnn_history['val_loss'], color="pink", label="CNN Validation Loss")

plt.title("Training History")
plt.ylabel("Loss")
plt.xlabel("Training Epoch")
plt.legend()
plt.savefig("resnet-historydo.png")
#plt.show()


plt.figure()

