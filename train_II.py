import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scipy.misc

import datetime
import os
import numpy as np
from numpy import newaxis


import tensorflow as tf

from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model

from models import create_auto_encoder
from params import args
from params import write_results


def r_squared(y_true, y_pred):
        from tensorflow.keras import backend as K
        SS_res = K.sum(K.square(y_true-y_pred))
        SS_tot = K.sum(K.square(y_pred - K.mean(y_pred)))
        return (1 - SS_res/(SS_tot + K.epsilon()))

def mean(y_true, y_pred):
        from tensorflow.keras import backend as K
        delta = K.abs(y_true - y_pred)
        delta = K.mean(delta)
        return delta

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


def LossCustom(alpha, beta):

        from tensorflow.keras import backend as K
        from tensorflow.keras import losses

        def loss(y_true, y_pred):

           ltot = losses.mean_absolute_error(y_true, y_pred)
           ltot = K.mean(ltot)


           pos = K.where(y_true>0.5)
           lh = losses.mean_absolute_error(K.gather_nd(y_true, pos), K.gather_nd(y_pred, pos))
           lh = K.mean(lh)

           pred = K.sum(y_pred, axis=1)
           targ = K.sum(y_true, axis=1)

           return ltot + alpha*lh

        return loss


def generator(n_batch):

         X = np.load("/DL/dl_coding/DL_code/Data/x.npy")
         ix = np.random.randint(0, X.shape[0], 500)
         xt = X[ix]
         del X
         Y = np.load("/DL/dl_coding/DL_code/Data/y.npy")
         yt = Y[ix]
         del Y
         idx = np.random.randint(0, xt.shape[0], n_batch)
         while True:
           batch_X, batch_Y = xt[idx], yt[idx]
           yield batch_X, batch_Y

def main():


    model = create_auto_encoder(filters=args.filters)
    model = multi_gpu_model(model, gpus=2)
    optimizer = Adam(0.0005)
    model.compile(optimizer=optimizer, loss=LossCustom(args.alpha, args.beta), metrics=[r_squared])


    folder_name = "/DL/dl_coding/DL_code/Res/" + str(args.results_path) + "/"
    os.makedirs(folder_name)
    trained_weights = folder_name + 'dl_fluids.h5'

    callbacks = [EarlyStopping(patience=20, verbose=1), ModelCheckpoint(trained_weights, verbose=1, save_best_only=True)]

    start_training = datetime.datetime.now()
    model.fit_generator(generator(args.batch_size), steps_per_epoch=int(2670/args.batch_size), epochs=args.epochs,verbose=1, callbacks=callbacks, \
                        validation_data=generator(200), validation_steps=(int(200/args.batch_size)))
    end_training = datetime.datetime.now()

    print(end_training - start_training)
    #scores = model.evaluate(X_test, Y_test, verbose=1)
    write_results(folder_name, (end_training - start_training))


if __name__ == '__main__':
    main()
           
