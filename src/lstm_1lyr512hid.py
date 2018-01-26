import numpy as np
import time
import random
import os
import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *

################################# Build RNN model ######################################
input_var = T.tensor3('inputs')
target_var = T.matrix('targets')
neg_var = T.matrix('neg')
(num_inputs, num_units, num_units_reg) =(4096, 512, 500)

print("Building model and compiling functions...")
l_in = InputLayer((None, None, num_inputs), input_var=input_var)
batchsize, seqlen, _ = l_in.input_var.shape
#genneric gate parameters
gate_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(gain=1.1),W_hid=lasagne.init.Orthogonal(gain=1.1),b=lasagne.init.Constant(0.))
#gate paramteres for forget gate
gate_parameters_forget = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(gain=1.1),W_hid=lasagne.init.Orthogonal(gain=1.1),b=lasagne.init.Constant(0.5))
#cell paramters
cell_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(gain=1.1),W_hid=lasagne.init.Orthogonal(gain=1.1),W_cell=None,b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.tanh)

l_lstm_1 = LSTMLayer(l_in,
        num_units,ingate=gate_parameters,forgetgate=gate_parameters_forget,cell=cell_parameters,outgate=gate_parameters,learn_init=True,grad_clipping=10.)


#Collapse-aggregate across temporal axis for Dense layer
l_shp = ReshapeLayer(l_lstm_1, (-1, num_units))
l_dense = DenseLayer(l_shp, num_units=num_units_reg,nonlinearity=None)
l_out = ReshapeLayer(l_dense, (batchsize, seqlen, num_units_reg))
network = l_out


######################## Creating train function #######################################
prediction = lasagne.layers.get_output(network)
prediction = prediction[0]
#compute cosine distance 
eps=1e-7
margin=0.10
lambda_=1.0
norm_in=T.sqrt(T.sum(prediction * prediction, axis=1))
norm_tar=T.sqrt(T.sum(target_var * target_var, axis=1))
norm_neg=T.sqrt(T.sum(neg_var * neg_var, axis=1))
#norm_in=input_var.sum(axis=1).reshape((input_var.shape[0], 1))
prod_xy_unnorm =  (prediction*target_var)
prod_xneg_unnorm =  (prediction*neg_var)

prod_xy_unnorm = prod_xy_unnorm.sum(axis=1)
prod_xneg_unnorm = prod_xneg_unnorm.sum(axis=1)

norm=norm_in*norm_tar + eps
norm_xneg=norm_in*norm_neg + eps

prod_xy= prod_xy_unnorm / (T.transpose(norm))
prod_xneg= prod_xneg_unnorm / (T.transpose(norm_xneg))
rank_loss=margin-prod_xy+prod_xneg
rank_loss= T.maximum(rank_loss,0)
rank_loss_m=T.mean(rank_loss,axis=0)

dist=1-prod_xy
dist_m=T.mean(dist,axis=0)
(lr,mtm)= (0.01,0.9)

#regularize all layers below dense
params = lasagne.layers.get_all_params(network, trainable=True)
loss = dist_m+lambda_*rank_loss_m
updates = lasagne.updates.adagrad(loss, params, learning_rate=lr)
train_fn = theano.function([input_var, target_var,neg_var], [loss,prediction],updates=updates)


######################## Creating test function #########################################
t_prediction = get_output(network, deterministic=True)
t_prediction = t_prediction[0]

val_fn = theano.function([input_var], [t_prediction])
