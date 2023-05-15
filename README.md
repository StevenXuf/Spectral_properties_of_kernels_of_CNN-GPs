# Spectral properties of kernels of CNN-GPs
This is my Master's project that studies the spectral properties (eigenvalues and eigenvectors) of Hilbert-Schmidt integral operator 
which is applied on Gaussian processes induced by neural networks with infinitely many neurons in hidden layers. For these who interests in the thesis, please see [here](https://mediatum.ub.tum.de/doc/1690738/kohrz68m4lxj2aho2nq3b30c2.pdf).

## 1. Preparation for the implementation of code
Our whole numerical experiments are programming by Python, and if you want to implement the codes here you must get Python environment installed in your computer. For the simplicity and consistency with the experiments in the thesis, we recommand that you install `Jupyter Notebook`. Please Use the following command:
```
pip3 install notebook
```
to install the notebook for both IOS and Linux systems if you did not install it yet.

After you installed the environment, you also need to install related libraries including: `matplotlib`, `numpy`, `torch`, `torchvision` and `cnn_gp`. To install the libraries, please use:
```
pip3 install [library_name]
```
for all libraries except for `cnn_gp`. Please visit [this page](https://github.com/waegemans/cnn-gp/tree/stable-backprop) to get the library `cnn_gp` which can compute the kernel matrix when we input some data, and follow the procedures that mentioned to install the library.

Presumably you have all the necessary libraries installed, please put the python modules `networks.py` and `eig_plot.py` under current path since you will need it when running the code.

## 2. Eigen-analysis
From here, we will do related analysis for kernel matrix, and concrete experiments can be found in the folder `Experiments`. 
### Import data
We are using MNIST dataset which consists of 60,000 training samples and 10,000 testing samples. We directly import MNIST data from `torchvision` using:
```python
import torchvision as tv

training_data=tv.datasets.MNIST('./mnist_train',train=True,download=True) #download training data
training_imgs=training_data.data #image data
training_labels=training_data.targets #image labels
```
Note that the image data must be normalized in range [0,1]:
```python
img_set=training_imgs[:100].reshape(100,1,28,28)/255 # normalize each pixles of 100 images into (0,1) 
```

### Defining neural networks as Gaussian processes
We need to define the network structures in order to compute kernel matrix of the network. Please follow:
```python
from networks import cnn, res_cnn

cnn3=cnn(3) #convolutional networks with 3 layers
res3=res_cnn(3) #residual convolutional networks with 3 layers
```

### Computing kernel matrix
To measure the similarity between any two images, their covariance must be computed using:
```python
cov_cnn=cnn3(img_set) #kernel matrix of CNN
cov_res=res3(img_set) #kernel matric of ResCNN
```

### Eigendecompostion
We do the eigendecomposition using `numpy` libray. Actually, there are 2 choices that could decompose the kernel matrix:
* numpy.linalg.eig
* numpy.linalg.svd

We use `svd` since it is numercially more stable than `eig`. If using `eig`, there will be sigularity in eigenvalues, so do not use it.
For instance, we use:
```python
imoprt numpy as np

U3_cnn,S3_cnn,V3_cnn=np.linalg.svd(cov_cnn)
```
to get the eigenvalues and eigenvectors of kernel matrix induced by a 3-layer CNN.

### Analysis for eigenvectors
In order to plot the pairwise graph of eigenvectors, we need to implement:
```python
from eig_plots import eigdecomp, plot_pairs, plot_eigvec_3d

n_samples=500  #define the size of image dataset
digits=[0,1] #define the digits we want to do analysis

#do the eigendecomposition for model cnn3 on digits 0 and 1 of total size 500 images
U3_cnn,S3_cnn,V3_cnn=eigdecomp(cnn3,digits,n_samples) 
plot_pairs(digits,U3_cnn,n_samples) #plot the top 5 pairwise eigenvectors
```
![pairwise plots of top 5 eigenvectors](/digit01cnn_3l.png)

which shows us the 2D graph above.

Implement
```python
plot_eigvec_3d(digits,U3_cnn,n_samples)
```
![3d plot of top 3 eigenvectors](/eigvec01cnn_3l.png)

to show the 3D graph for top 3 eigenvectors.

## 3. Network accuracy
We first talk the accuracy for different network depths. We define a function `model_acc(model,training_batch_ind)` to compute the network accuracy for different depths. Call function `model_acc` in file `Exp4_acc_of_depths.ipynb`
```python
model=cnn3
training_batch_ind=10 #index of training batch
acc_cnn3=model_acc(model,training_batch_ind) #return average accuracy trained on a batch and whole testing data
```
to get averge accuracy on a specific training batch.

Likewise, we also define another function named `model_acc()` to measure the influence of kernel sizes on classifcation accuracy in `Exp5_acc_of_kernel_sizes.ipynb`. See the files for more details.
