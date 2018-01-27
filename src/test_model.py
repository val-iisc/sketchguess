import math
import random
import numpy.random as random_np
import numpy as np
import lasagne
import scipy.io
import time
import os

print('Initializing LSTM model...')
execfile('src/lstm_1lyr512hid.py')

with np.load('models/lstm_model_1lyr512hid_earlystopped.npz') as f:
         param_values = [f['arr_%d' % i] for i in range(len(f.files))]
         lasagne.layers.set_all_param_values(network, param_values)

# Testing starts here
start_time = time.time()
te_size_counter = 0
print "[Test] data/cnn_features_test.npz"
arr = np.load("data/cnn_features_test.npz")
x_test_chunk, _ = np.array(arr['arr_0']),np.array(arr['arr_1'])
test_pred=[]
for batch in range(len(x_test_chunk)):
    img_seq = x_test_chunk[batch].astype('float32')
    length_seq = img_seq.shape[0]
    res = val_fn([img_seq])
    te_pred = res[0]
    te_size_counter = te_size_counter + 1
    test_pred.append(te_pred)
np.savez("out/pred_release.npz",test_pred)
print("Testing completed in %d seconds....." %(time.time()-start_time))
