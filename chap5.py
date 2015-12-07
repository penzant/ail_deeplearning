# coding:utf-8
### Stacked auto-encoderの実装

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
rng = np.random.RandomState(1234)

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist_x, mnist_y = mnist.data.astype("float32")/255.0, mnist.target.astype("int32")

#Autoencoder
class Autoencoder:
    def __init__(self,visible_dim,hidden_dim,W,function):
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.W = W
        self.function = function
                
        self.a = theano.shared(np.zeros(visible_dim).astype(np.float32),name="a")
        self.b = theano.shared(np.zeros(hidden_dim).astype(np.float32),name="b")
        self.params = [self.W,self.a,self.b]
        
    #encoder
    def encode(self,x):
        u = T.dot(x, self.W)+self.b
        y = self.function(u)
        return y
    
    #decoder
    def decode(self,x):
        u = T.dot(x, self.W.T)+self.a
        y = self.function(u)
        return y
    
    #forward propagation
    def prop(self,x):
        y = self.encode(x)
        reconst_x = self.decode(y)
        return reconst_x
    
    #reconstruction error
    def reconst_error(self,x,noise):
        tilde_x = x*noise
        reconst_x = self.prop(tilde_x)
        error = T.mean(T.sum(T.nnet.binary_crossentropy(reconst_x,x),axis=1))
        return error, reconst_x
        
#SGD
def sgd(params,gparams,lr=0.1): 
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        updates[param] = param - np.float32(lr) * gparam
    return updates

def AdaGrad(params, gparams, lr=0.01):
    updates = OrderedDict()
    sumgrads = [theano.shared(np.zeros(p.shape.eval()).astype('float32')) for p in params]
    for param, gparam, sumgrad in zip(params, gparams, sumgrads):
        sgrad = sumgrad  + gparam * gparam
        param_diff = - (np.float32(lr) / T.sqrt(sgrad + np.float32(1.e-6))) * gparam
        updates[param] = param + param_diff
        updates[sumgrad] = sgrad
    return updates

#Multi Layer Perceptron
class Layer:
    def __init__(self, in_dim, out_dim, function):
        self.W = theano.shared(rng.uniform(low=-np.sqrt(6./(in_dim+out_dim)),
                                           high=np.sqrt(6./(in_dim+out_dim)),
                                           size=(in_dim,out_dim)).astype('float32'), name="W")
        self.b = theano.shared(np.zeros(out_dim).astype("float32"), name="bias")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.function = function
        self.params = [self.W, self.b]
        self.set_pretraining()

    def fprop(self, x):
        h = self.function(T.dot(x, self.W) + self.b)
        self.h = h
        return h
    
    def set_pretraining(self):
        ae = Autoencoder(self.in_dim,self.out_dim,self.W,self.function)

        x = T.fmatrix('x')
        noise = T.fmatrix('noise')

        cost,reconst_x = ae.reconst_error(x,noise)
        params  = ae.params
        gparams = T.grad(cost, params)
        updates = sgd(params,gparams)

        self.pretraining = theano.function([x,noise], [cost,reconst_x], updates=updates, allow_input_downcast=True)
        
        hidden = ae.encode(x)
        self.encode_function = theano.function([x], hidden, allow_input_downcast=True)

train_X, valid_X, train_y, valid_y = train_test_split(mnist_x, mnist_y, test_size=0.2, random_state=42)

activation = T.nnet.sigmoid
layers = [
    Layer(784, 500, activation),
    Layer(500, 500, activation),
    Layer(500, 500, activation),
    Layer(500, 10, T.nnet.softmax)
]
corruption_level = np.float32(0.3)

#Pre-training
X = train_X
for l, layer in enumerate(layers[:-1]):
    batch_size = 100
    nbatches = X.shape[0] // batch_size

    for epoch in range(10):
        X = shuffle(X)
        err_all=[]
        for i in range(0,nbatches):
            start = i * batch_size
            end   = start + batch_size

            noise = rng.binomial(size=X[start:end].shape, n=1, p=1-corruption_level)
            err,reconst_x = layer.pretraining(X[start:end],noise)
            err_all.append(err)
        print "Pre-training:: layer:%d, Epoch:%d, Error:%lf" %(l,epoch, np.mean(err_all))
    X = layer.encode_function(X)

#Fine-tuning
x, t = T.fmatrix("x"), T.ivector("t")
params = []
for i, layer in enumerate(layers):
    params += layer.params
    if i == 0:
        layer_out = layer.fprop(x)
    else:
        layer_out = layer.fprop(layer_out)

y = layers[-1].h
cost = - T.mean((T.log(y))[T.arange(x.shape[0]), t])

gparams = T.grad(cost, params)
#updates = sgd(params,gparams)
updates = AdaGrad(params,gparams,0.01)

train = theano.function([x,t], cost, updates=updates)
valid  = theano.function([x,t],[cost, T.argmax(y, axis=1)])
test  = theano.function([x],T.argmax(y, axis=1))

batch_size = 100
nbatches = train_X.shape[0]//batch_size
for epoch in range(50):
    train_X, train_y = shuffle(train_X, train_y)
    for i in range(nbatches):
        start = i * batch_size
        end = start + batch_size

        train(train_X[start:end], train_y[start:end])
    valid_cost, pred = valid(valid_X, valid_y)
    print "EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f"%(epoch+1, valid_cost, f1_score(valid_y, pred, average="macro"))

#pred_y = test(test_X)
