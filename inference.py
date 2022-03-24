import numpy as np

from keras.models import Model, load_model
from keras.utils import multi_gpu_model


### import the architecture and the arguments
from models import create_auto_encoder
from params import args


#import the architecture 
#and transform it to accept multi gpu 
model = create_auto_encoder(filters=args.filters)
model = multi_gpu_model(model, gpus=2)

#the path where the .h5 is saved
path = "/home/where_the_.h5_is"
model.load_weights(path+"dl_fluids.h5")

#load the input 
x_test = np.load("/home/data/")
x_test = np.expand_dims(x_test[1784,:,:,:], axis=0)

#loop through the data
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
