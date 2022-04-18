import matplotlib.pyplot as plt
import numpy as np
import torchvision as tv

data=tv.datasets.MNIST('./mnist_train',train=True,download=True).data
labels=tv.datasets.MNIST('./mnist_train',train=True,download=True).targets

colors={0:'r',1:'b',2:'y',3:'g'}

def digit_ind(digits,n_samples=500,start=0):
    ind=[]
    for i in range(start,60000):
        if labels[i] in digits:
            ind.append(i)
        if len(ind)==n_samples:
            break
    return ind
    
def plot_eigvec_2d(digits,eigvec1,eigvec2,n_samples=500,start=0):
    ind=digit_ind(digits,n_samples,start)
    for digit in digits:
        c=[colors[digits.index(digit)] for i in range(sum(labels[ind]==digit))]
        plt.scatter(eigvec1[labels[ind]==digit],eigvec2[labels[ind]==digit],
                    c=c,label=f'{digit}')
    #plt.xlabel(r'$\phi_1$',fontsize=12)
    #plt.ylabel(r'$\phi_2$',fontsize=12)
    plt.legend()
    
def plot_eigvec_3d(digits,eigvec,n_samples=500,view=(18,155),start=0):
    fig = plt.figure(figsize=(6,6),dpi=80)
    ax = fig.add_subplot(projection='3d')
    
    ind=digit_ind(digits,n_samples,start)
    for digit in digits:
        c=[colors[digits.index(digit)] for i in range(sum(labels[ind]==digit))]
        ax.scatter(eigvec[:,0][labels[ind]==digit],eigvec[:,1][labels[ind]==digit],
                   eigvec[:,2][labels[ind]==digit],c=c,label=f'{digit}')
    ax.set_xlabel(r'$\phi_0$',fontsize=12)
    ax.set_ylabel(r'$\phi_1$',fontsize=12)
    ax.set_zlabel(r'$\phi_2$',fontsize=12)
    ax.view_init(*view)
    plt.legend()
    
def eigdecomp(model,digits,n_samples=500,start=0):
    ind=digit_ind(digits,n_samples,start)
    test_data=data[ind].view(n_samples,1,28,28)/255
    k=model(test_data)
    U,S,V=np.linalg.svd(k)
    return U,S,V
    
def plot_pairs(digits,eigvec,n_samples=500,pairs=4,bench=0):
    for i in range(1,pairs+1):
        plt.subplot(pairs//2,2,i)
        plot_eigvec_2d(digits,eigvec[:,bench],eigvec[:,i],n_samples)
        plt.title(r'$\phi_{%s}$ vs $\phi_{%s}$'%(bench,i))
    plt.tight_layout()
    
def plot_eigval(eigval):
    plt.plot(np.log(eigval))
    
def plot_more(digits,eigvec,n_samples=500,pairs=4,bench=0):
    step=int(n_samples/pairs)
    for i in range(1,pairs+1):
        plt.subplot(pairs//2,2,i)
        plot_eigvec_2d(digits,eigvec[:,bench],eigvec[:,i*step-1],n_samples)
        plt.title(r'$\phi_{%s}$ vs $\phi_{%s}$'%(bench,i*step-1))
    plt.tight_layout()