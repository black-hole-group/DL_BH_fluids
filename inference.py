from keras.models import Model, load_model
import numpy as np
from models import create_auto_encoder
from params import args
from keras.utils import multi_gpu_model

import os

model = create_auto_encoder(filters=args.filters)
model = multi_gpu_model(model, gpus=2)
path = "/DL/dl_coding/DL_code/Results/CNN64_20/"
model.load_weights(path+"dl_fluids.h5")

x_test = np.load("/DL/dl_coding/DL_code/Data/x.npy")
#x_test = np.expand_dims(x_test[int(1892+236),:,:,:], axis=0)
x_test = np.expand_dims(x_test[1784,:,:,:], axis=0)

#r = 2366 - 1784
for i in range(100):

       if i == 0:
          preds = model.predict(x_test)
          del x_test

       else:
          x = np.load(path + str(i) +".npy")
          preds = model.predict(x)
         
       preds= np.asarray(preds)
       np.save(path+ str(i+1) + ".npy", preds)
       print(preds.shape)
