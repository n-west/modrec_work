
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle

colors = ["red", "green", "blue", "yellow", "purple", "black", "orange", "grey", "lawngreen", "cyan", "khaki"]

# Plot the number of filters / feature maps
c1 = cPickle.load(open("conv-radio-modrec-20filts_2_layers:v0:56.950", "rb"))
c2 = cPickle.load(open("conv-radio-modrec-30filts_2_layers:v0:58.398", "rb"))
c3 = cPickle.load(open("conv-radio-modrec-40filts_2_layers:v0:56.684", "rb"))
c4 = cPickle.load(open("conv-radio-modrec-50filts_2_layers:v0:58.941", "rb"))
c5 = cPickle.load(open("conv-radio-modrec-60filts_2_layers:v0:56.650", "rb"))
c6 = cPickle.load(open("conv-radio-modrec-70filts_2_layers:v0:55.459", "rb"))
c7 = cPickle.load(open("conv-radio-modrec-80filts_2_layers:v0:55.436", "rb"))
c8 = cPickle.load(open("conv-radio-modrec-90filts_2_layers:v0:56.550", "rb"))

plt.scatter(c1[0], c1[1], color="red", label="20 filters")
plt.scatter(c2[0], c2[1], color="green", label="30 filters")
plt.scatter(c3[0], c3[1], color="blue", label="40 filters")
plt.scatter(c4[0], c4[1], color="purple", label="50 filters")
plt.scatter(c5[0], c5[1], color="yellow", label="60 filters")
plt.scatter(c6[0], c6[1], color="black", label="70 filters")
plt.scatter(c7[0], c7[1], color="orange", label="80 filters")
plt.scatter(c8[0], c8[1], color="cyan", label="90 filters")
plt.title("Accuracy vs SNR for nfilts per layer")
plt.ylabel("Accuracy [top-1]")
plt.xlabel("SNR [dB]")
plt.legend()
plt.savefig("nfilts-hyperopt.png")
#plt.show()


plt.figure()

files=["conv-radio-modrec-50filts_12taps_2_layers:v0:61.355",
"conv-radio-modrec-50filts_11taps_2_layers:v0:61.080",
"conv-radio-modrec-50filts_10taps_2_layers:v0:59.811",
"conv-radio-modrec-50filts_9taps_2_layers:v0:61.475",
"conv-radio-modrec-50filts_8taps_2_layers:v0:61.314",
"conv-radio-modrec-50filts_7taps_2_layers:v0:59.936",
"conv-radio-modrec-50filts_6taps_2_layers:v0:58.766",
"conv-radio-modrec-50filts_5taps_2_layers:v0:60.464",
"conv-radio-modrec-50filts_3taps_2_layers:v0:57.909",
"conv-radio-modrec-50filts_2taps_2_layers:v0:56.343"]

labels=["12 taps",
"11 taps",
"10 taps",
"9 taps",
"8 taps",
"7 taps",
"6 taps",
"5 taps",
"4 taps",
"3 taps",
"2 taps"]

counter = 0
for f in files:
    c = cPickle.load(open(f, "rb"))
    plt.scatter(c[0], c[1], color=colors[counter], label=labels[counter])
    counter += 1
plt.title("Accuracy for 2-conv layer network with 50 filters")
plt.legend()
plt.ylabel("Accuracy [top-1]")
plt.xlabel("SNR [dB]")
plt.savefig("ntaps-hyperopt.png")


plt.figure()

files = [
"conv-radio-modrec-50filts_8taps_6_layers:v0:59.407",
"conv-radio-modrec-50filts_8taps_5_layers:v0:59.034",
"conv-radio-modrec-50filts_8taps_4_layers:v0:59.257",
"conv-radio-modrec-50filts_8taps_3_layers:v0:61.002",
"conv-radio-modrec-50filts_8taps_2_layers:v0:61.314"]
labels = [
"6 layers",
"5 layers",
"4 layers",
"3 layers",
"2 layers"]

counter = 0
for f in files:
    c = cPickle.load(open(f, "rb"))
    plt.scatter(c[0], c[1], color=colors[counter], label=labels[counter])
    counter += 1
plt.title("Accuracy for CNN with 50 8-tap filters per conv layer")
plt.legend()
plt.ylabel("Accuracy [top-1]")
plt.xlabel("SNR [dB]")
plt.savefig("layers-hyperopt.png")
#plt.show()
