import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scipy.misc

import datetime
import os
import numpy as np
from numpy import newaxis

from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model

#from data import get_data
from models import create_auto_encoder
from params import args
from params import write_results
#from CyclicLearningRate import CyclicLR

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def r_squared(y_true, y_pred):
	from tensorflow.keras import backend as K
	SS_res = K.sum(K.square(y_true-y_pred))
	SS_tot = K.sum(K.square(y_pred - K.mean(y_pred)))
	return (1 - SS_res/(SS_tot + K.epsilon()))


def LossCustom(alpha, beta, delta):

        from tensorflow.keras import backend as K
        from tensorflow.keras import losses

        def loss(y_true, y_pred):

           ltot = losses.mean_absolute_error(y_true, y_pred)
           ltot = K.mean(ltot)


           pos = K.where(y_true>beta)
           lh = losses.mean_absolute_error(K.gather_nd(y_true, pos), K.gather_nd(y_pred, pos))
           lh = K.mean(lh)

   
           posi = K.where(y_true<beta)
           ll = losses.mean_absolute_error(K.gather_nd(y_true, pos), K.gather_nd(y_pred, pos))
           ll = K.mean(ll)

           l3 = 0.3*losses.mean_absolute_error(y_true[:,:22,:,:], y_pred[:,:22, :,:])
           l3 = K.mean(l3)

           l4 = 0.3*losses.mean_absolute_error(y_true[:,22:, 25:166,:], y_pred[:,22:, 25:166, :])
           l4 = K.mean(l4)

           #density_tot_t = tf.keras.backend.sum(y_true, axis=1)
           #density_tot_p = tf.keras.backend.sum(y_pred, axis=1)


           #mass = losses.mean_absolute_error(tf.keras.backend.sum(y_true, axis=1), tf.keras.backend.sum(y_pred, axis=1)) 
           #mass = K.mean(mass)

           return ltot + alpha*lh + delta*ll + l3 + l4


        return loss


def CustomLoss():
	from tensorflow.keras import backend as K
	from tensorflow.keras import losses

	def LossFunc(y_true, y_pred):
		#total   
		l1 = losses.mean_squared_error(y_true, y_pred)
		l1 = K.mean(l1)
 
		#high density 2 5
		l2 = 8*losses.mean_absolute_error(y_true[:,56:186, 35:156,:], y_pred[:,56:186, 35:156,:]) 
		l2 = K.mean(l2)

		#accretion_disk 2 2
		l3 = 5*losses.mean_absolute_error(y_true[:,:22,:,:], y_pred[:,:22, :,:])
		l3 = K.mean(l3)

		#torus 5 10
		l4 = 10*losses.mean_absolute_error(y_true[:,45:, 25:166,:], y_pred[:,45:, 25:166, :])
		l4 = K.mean(l4)

		#diffusion 5
		l5 = 4*losses.mean_absolute_error(y_true[:,22:, 5:186,:], y_pred[:,22:, 5:186,:])
		l5 = K.mean(l5)

		loss = l1 + l2 + l3 + l4 + l5

		return loss

	return LossFunc

def main():
    X = np.load("/DL/dl_coding/DL_code/Data/x.npy")
    Y = np.load("/DL/dl_coding/DL_code/Data/y.npy")
 	
    size_train = int(0.8*X.shape[0])
    size_val = int(0.1*X.shape[0])
    
    print(size_train)
    print(size_val)
    
    print(X[:size_train].shape)
    print(Y[:size_train].shape)
    #print(mass[:size_train].shape)
    model = create_auto_encoder(filters=args.filters)

    model = multi_gpu_model(model, gpus=2)

    folder_name = '/DL/dl_coding/DL_code/Results/' + args.results_path
    os.makedirs(folder_name)
    trained_weights = folder_name + 'dl_fluids.h5'

    optimizer = Adam(0.0005)
    
    callbacks = [EarlyStopping(patience=20, verbose=1), ModelCheckpoint(trained_weights, verbose=1, save_best_only=True)]

    model.compile(optimizer=optimizer, loss={'prediction': CustomLoss()}, metrics=[r_squared])

    start_training = datetime.datetime.now()
    history = model.fit(X[:size_train], Y[:size_train], epochs=args.epochs, batch_size=args.batch_size, validation_data=(X[size_train:int(size_train+size_val)], Y[size_train:int(size_train+size_val)]), callbacks=callbacks)
    end_training = datetime.datetime.now()
    
    save_loss = open(folder_name + "loss_values.txt", "w")
    save_loss.write(str(history.history['loss']) + "\n")
    save_loss.write(str(history.history['val_loss']) + "\n")
    
    #scores = model.evaluate(X[int(size_train+size_val):], Y[int(size_train+size_val):], verbose=1)
 

if __name__ == '__main__':
    main()
