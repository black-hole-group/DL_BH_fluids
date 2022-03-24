# DL_BH_fluids
This is the model we proposed in Duarte et al. 2022. All the details are available and described in the paper. 

model.py: the architecture we used is based on a U-Net with modifications in the number of layers and the input/output. 

The main difference is that our input accepts the temporal dimension with two-spatial dimensions, while the classical U-Net accepts only the spatial dimensions. 

params.py: hyperparameters you can change before training the model. 

  epochs: the number of epochs (default = 100)
  
  batch_size: the size of the mini-batch (default = 64)
  
  filters: how many filters are in the first layer (default = 32)
  
  results_path: the path where you will save the training 
  
  alpha, beta, delta, gamma: loss function parameters (default = 0.1)
  

train.py: training settings without generator

train_gen.py: training setting with a generator to save memory

