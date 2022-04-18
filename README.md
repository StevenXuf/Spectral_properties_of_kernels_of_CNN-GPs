# Spectral properties of kernels of CNN-GP
This is my Master's project that studies the spectral properties (eigenvalues and eigenvectors) of Hilbert-Schmidt integral operator 
which was applied on Gaussian processes induced by neural networks with infinitely many neurons in hidden layers.

## Preparation for the implementation of code
Our whole numerical experiments are programming by Python, and if you want to implement the codes here you must get Python environment installed in your computer. For the simplicity and consistency with the experiments in the thesis, we recommand that you install `Jupyter Notebook`. Please Use the following command:
```
pip3 install notebook
```
to install the notebook for both Mac and Linux systems if you did not install it yet.

After you installed the environment, please visit [this page](https://github.com/waegemans/cnn-gp/tree/stable-backprop) to get the library that can compute the kernel matrix when we input some data, and follow the procedures that mentioned to install the library.
