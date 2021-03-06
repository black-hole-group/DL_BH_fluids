Black Hole Weather Forecasting with Deep Learning: A Pilot Study
==============================================

This is the deep learning model that we proposed in [*Black Hole Weather Forecasting with Deep Learning: A Pilot Study* (Duarte, Nemmen & Navarro, MNRAS, in press)](https://arxiv.org/abs/2102.06242). All the details are available and described in the paper. This repository includes the following files:

We trained the model using Tensorflow 1.8.0 with multi_gpu (P6000 and GP100).

`model.py`: the architecture we used is based on a U-Net with modifications in the number of layers and the input/output. 

The main difference is that our input accepts the temporal dimension with two-spatial dimensions, while the classical U-Net accepts only the spatial dimensions. 

The input is a tensor `(N, 256, 192, 5)`.

`params.py`: hyperparameters you can change before training the model, which are the following: 

- epochs: the number of epochs (default = 100)
- batch_size: the size of the mini-batch (default = 64)
- filters: how many filters are in the first layer (default = 32)
- results_path: the path where you will save the training 
- alpha, beta, delta, gamma: loss function parameters (default = 0.1)
  
`train.py`: training settings without generator

`train_gen.py`: training setting with a generator to save memory

`inference.py`: how to create predictions using .h5 file

# Data files

You can find the data files with the trained neural network weights in our [group's data repository](https://doi.org/10.6084/m9.figshare.19412147.v1). They are in the HDF5 binary format, following Tensor Flow standards. You can load the weights using 

    model.load("/path/to/.h5/dl_fluids.h5")
