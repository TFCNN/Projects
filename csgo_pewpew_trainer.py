# James William Fletcher (james@voxdsp.com)
#   - CS:GO PewPew Trigger Bot Weight Trainer
#   https://github.com/tfcnn
#       June 2021
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from random import seed
from random import randbytes
from hashlib import sha256
from time import time_ns
from sys import exit
from os.path import isdir
from os import mkdir
from tensorflow.keras import backend as K

# hyperparameters
seed(8008135)
project = "aim_model"
training_iterations = 333
activator = 'tanh'
# layers = 3
layer_units = 1
batches = 24

tc = 302   # target sample count/length
ntc = 466  # non-target sample count/length

# helpers (https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison)
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

# load training data
nontargets_x = []
nontargets_y = np.zeros([ntc, 1], dtype=np.float32)
with open("nontargets.dat", 'rb') as f:
    data = np.fromfile(f, dtype=np.float32)
    nontargets_x = np.reshape(data, [ntc, 2352])

targets_x = []
targets_y = np.ones([tc, 1], dtype=np.float32)
with open("targets.dat", 'rb') as f:
    data = np.fromfile(f, dtype=np.float32)
    targets_x = np.reshape(data, [tc, 2352])

# print(targets_y.shape)
# print(nontargets_y.shape)
# exit()

train_x = np.concatenate((nontargets_x, targets_x), axis=0)
train_y = np.concatenate((nontargets_y, targets_y), axis=0)

shuffle_in_unison(train_x, train_y)

x_val = train_x[-230:]
y_val = train_y[-230:]
x_train = train_x[:-230]
y_train = train_y[:-230]

# print(x_val.shape)
# print(y_val.shape)
# print(x_train.shape)
# print(y_train.shape)
# exit()

# print(y_train)
# exit()


# construct neural network
model = Sequential()
model.add(Dense(layer_units, activation=activator, input_dim=2352))
# for x in range(layers-2):
#     model.add(Dense(layer_units, activation=activator))
model.add(Dense(1, activation='sigmoid'))

# optim = keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer='adam', loss='mean_squared_error')


# train network
st = time_ns()
model.fit(train_x, train_y, epochs=training_iterations, batch_size=batches)
# model.fit(x_train, y_train, epochs=training_iterations, batch_size=batches, validation_data=(x_val, y_val))
timetaken = (time_ns()-st)/1e+9
print("")
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")


# save model
if not isdir(project):
    mkdir(project)
if isdir(project):
    # save model
    f = open(project + "/model.txt", "w")
    if f:
        f.write(model.to_json())
    f.close()

    # save HDF5 weights
    model.save_weights(project + "/weights.h5")

    # save flat weights
    for layer in model.layers:
        if layer.get_weights() != []:
            np.savetxt(project + "/" + layer.name + ".csv", layer.get_weights()[0].flatten(), delimiter=",") # weights
            np.savetxt(project + "/" + layer.name + "_bias.csv", layer.get_weights()[1].flatten(), delimiter=",") # bias

    # save weights for C array
    print("")
    print("Exporting weights...")
    li = 0
    f = open(project + "/" + project + "_layers.h", "w")
    f.write("#ifndef " + project + "_layers\n#define " + project + "_layers\n\n")
    if f:
        for layer in model.layers:
            total_layer_weights = layer.get_weights()[0].flatten().shape[0]
            total_layer_units = layer.units
            layer_weights_per_unit = total_layer_weights / total_layer_units
            #print(layer.get_weights()[0].flatten().shape)
            #print(layer.units)
            print("+ Layer:", li)
            print("Total layer weights:", total_layer_weights)
            print("Total layer units:", total_layer_units)
            print("Weights per unit:", int(layer_weights_per_unit))

            f.write("const float " + project + "_layer" + str(li) + "[] = {")
            isfirst = 0
            wc = 0
            bc = 0
            if layer.get_weights() != []:
                for weight in layer.get_weights()[0].flatten():
                    wc += 1
                    if isfirst == 0:
                        f.write(str(weight))
                        isfirst = 1
                    else:
                        f.write("," + str(weight))
                    if wc == layer_weights_per_unit:
                        f.write(", /* bias */ " + str(layer.get_weights()[1].flatten()[bc]))
                        #print("bias", str(layer.get_weights()[1].flatten()[bc]))
                        wc = 0
                        bc += 1
            f.write("};\n\n")
            li += 1
    f.write("#endif\n")
    f.close()


# show results
print("")
pt = model.predict(targets_x)
ptavg = np.average(pt)

pnt = model.predict(nontargets_x)
pntavg = np.average(pnt)

cnzpt =  np.count_nonzero(pt <= pntavg)
cnzpts = np.count_nonzero(pt >= ptavg)
avgsuccesspt = (100/tc)*cnzpts
avgfailpt = (100/tc)*cnzpt
outlierspt = tc - int(cnzpt + cnzpts)

cnzpnt =  np.count_nonzero(pnt >= ptavg)
cnzpnts = np.count_nonzero(pnt <= pntavg)
avgsuccesspnt = (100/ntc)*cnzpnts
avgfailpnt = (100/ntc)*cnzpnt
outlierspnt = ntc - int(cnzpnts + cnzpnt)

print("training_iterations:", training_iterations)
print("activator:", activator)
# print("layers:", layers)
print("layer_units:", layer_units)
print("batches:", batches)
print("")
print("target:", "{:.0f}".format(np.sum(pt)) + "/" + str(tc))
print("target-max:", "{:.3f}".format(np.amax(pt)))
print("target-avg:", "{:.3f}".format(ptavg))
print("target-min:", "{:.3f}".format(np.amin(pt)))
print("target-avg-success:", str(cnzpts) + "/" + str(tc), "(" + "{:.2f}".format(avgsuccesspt) + "%)")
print("target-avg-fail:", str(cnzpt) + "/" + str(tc), "(" + "{:.2f}".format(avgfailpt) + "%)")
print("target-avg-outliers:", str(outlierspt) + "/" + str(tc), "(" + "{:.2f}".format((100/tc)*outlierspt) + "%)")
print("")
print("nontarget:", "{:.0f}".format(np.sum(pnt)) + "/" + str(ntc))
print("nontarget-max:", "{:.3f}".format(np.amax(pnt)))
print("nontarget-avg:", "{:.3f}".format(pntavg))
print("nontarget-min:", "{:.3f}".format(np.amin(pnt)))
print("nontarget-avg-success:", str(cnzpnts) + "/" + str(ntc), "(" + "{:.2f}".format(avgsuccesspnt) + "%)")
print("nontarget-avg-fail:", str(cnzpnt) + "/" + str(ntc), "(" + "{:.2f}".format(avgfailpnt) + "%)")
print("nontarget-avg-outliers:", str(outlierspnt) + "/" + str(ntc), "(" + "{:.2f}".format((100/ntc)*outlierspnt) + "%)")