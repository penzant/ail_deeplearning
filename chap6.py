
# coding: utf-8
# CNN for MNIST

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

from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

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
    
class Layer:
    def __init__(self, in_dim, out_dim, function, use_dropout=False, drop_rate=0.5):
        self.W = theano.shared(rng.uniform(low=-np.sqrt(6./(in_dim+out_dim)),
                                           high=np.sqrt(6./(in_dim+out_dim)),
                                           size=(in_dim,out_dim)).astype('float32'), name="W")
        self.b = theano.shared(np.zeros(out_dim).astype("float32"), name="bias")
        self.function = function
        self.params = [self.W, self.b]
        if use_dropout and not testing:
            self.drop = Dropout(in_dim, drop_rate)
            self.mask = self.drop.mask
            self.use_dropout = True
        else:
            self.use_dropout = False

    def fprop(self, x):
        if self.use_dropout:
            h = self.function(T.dot(self.mask * x, self.W) + self.b)
        else:
            h = self.function(T.dot(x, self.W) + self.b)
        self.h = h
        return h
    
class Conv:
    def __init__(self,filter_shape,function,border_mode="valid",subsample=(1, 1)):
        
        self.function = function
        self.border_mode = border_mode
        self.subsample = subsample
        
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
        
        self.W = theano.shared(rng.uniform(
                    low=-4*np.sqrt(6. / (fan_in + fan_out)),
                    high=4*np.sqrt(6. / (fan_in + fan_out)),
                    size=filter_shape
                ).astype("float32"),name="W")
        self.b = theano.shared(np.zeros((filter_shape[0],), dtype="float32"),name="b")
        self.params = [self.W,self.b]
        
    def fprop(self,x):
        conv_out = conv.conv2d(x,self.W,
                               border_mode=self.border_mode,
                               subsample=self.subsample)
        y = self.function(conv_out + self.b[np.newaxis,:,np.newaxis,np.newaxis])
        return y


class Pooling:
    def __init__(self,pool_size=(2,2)):
        self.pool_size=pool_size
        self.params = []
    def fprop(self,x):
        return downsample.max_pool_2d(x,self.pool_size,ignore_border=True)

    
class Flatten:
    def __init__(self,outdim=2):
        self.outdim = outdim
        self.params = []
    def fprop(self,x):
        return T.flatten(x,self.outdim)


train_X, valid_X, train_y, valid_y = train_test_split(mnist_x, mnist_y, test_size=0.2, random_state=42)

activation = T.nnet.sigmoid
layers = [
    Conv((20,1,5,5),activation),
    Pooling((2,2)),
    Conv((50,20,5,5),activation),
    Pooling((2,2)),
    Flatten(2),
    Layer(800,500, activation),#800=((((28-5+1)/2)-5+1)/2)**2*50
    Layer(500,10, T.nnet.softmax)
]

x, t = T.fmatrix("x"), T.ivector("t")
x_4d = x.reshape((x.shape[0],1,28,28))

params = []
layer_out = x_4d
for i, layer in enumerate(layers):
    params += layer.params
    layer_out = layer.fprop(layer_out)

y = layers[-1].h
cost = - T.mean((T.log(y))[T.arange(x.shape[0]), t])

gparams = T.grad(cost, params)
# updates = sgd(params,gparams)
updates = AdaGrad(params,gparams)

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
        if i % 100 ==0 : print "batches:{0}/{1}".format(i,nbatches)
    valid_cost, pred = valid(valid_X, valid_y)
    print "EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f"%(epoch+1, valid_cost, f1_score(valid_y, pred, average="macro"))

