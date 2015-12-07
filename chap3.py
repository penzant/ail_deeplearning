# coding: utf-8

### Multilayer perceptron for MNIST

import numpy as np
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score

np.random.seed(1234)

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

def sigmoid(x):
    return 1/(1+np.exp(-x))
def diff_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1)[:,np.newaxis]
def diff_softmax(x):
    return softmax(x)*(np.ones(x.shape)-softmax(x))
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def diff_tanh(x):
    return 1-tanh(x)**2
def ReLU(x):
    return max(0,x)

class Layer:
    def __init__(self,in_dim,out_dim,function,diff_function):
        #Xavier
        self.W = np.random.uniform(
                                    low=-np.sqrt(6./(in_dim+out_dim)), 
                                    high=np.sqrt(6./(in_dim+out_dim)), 
                                    size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.function = function
        
        self.diff_function = diff_function
        self.u     = None
        self.delta = None

    #foward propagation
    def fprop(self,x):
        u = np.dot(x,self.W)+self.b
        z = self.function(u)
        self.u = u
        return z

    #back propagation
    def bprop(self,delta,W):
        delta = self.diff_function(self.u)*np.dot(delta,W.T)
        self.delta = delta
        return delta

def fprops(layers, x):
    z = x
    for layer in layers:
        z = layer.fprop(z)
    return z
def bprops(layers, delta):
    for i,layer in enumerate(layers[::-1]):
        if i==0:
            layer.delta = delta
        else:
            delta = layer.bprop(delta,_W)
        _W = layer.W

def one_of_k(n):
    return [0 if x != n else 1 for x in range(10)]

X, Y = shuffle(mnist.data, mnist.target)
X = X / 255.0
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.2)

train_y = np.asarray([one_of_k(x) for x in train_y])

layers = [Layer(len(train_X[0]),100,sigmoid,diff_sigmoid),
          Layer(100,len(train_y[0]),softmax,diff_softmax)]

def train(X,d,eps=1):
    #forward propagation
    y = fprops(layers,X)
        
    #cost function & delta
    cost = np.sum(-d*np.log(y)-(1-d)*np.log(1-y))
    delta = y-d
    
    #back propagation
    bprops(layers,delta)

    #update parameters
    z = X
    for layer in layers:
        dW = np.dot(z.T,layer.delta)
        db = np.dot(np.ones(len(z)),layer.delta)

        layer.W = layer.W - eps*dW
        layer.b = layer.b - eps*db

        z = layer.fprop(z)
        
    #train cost
    y = fprops(layers,X)
    cost = np.sum(-d*np.log(y)-(1-d)*np.log(1-y))
    
    return cost

def test(X):
    y = fprops(layers,X)
    y = np.asarray([np.argmax(x) for x in y])
    return y

for epoch in range(100):
    train_X, train_y = shuffle(train_X, train_y)
    for x, y in zip(train_X, train_y):
         train(x[np.newaxis,:],y[np.newaxis,:],0.01)
    pred_y = test(test_X)
    f1_pred = f1_score(test_y, pred_y, average='micro')
    print "epoc[%03d]:%s" % (epoch, f1_pred)


