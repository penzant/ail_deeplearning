# coding: utf-8

# Multi Layer Perceptron using theano

from collections import OrderedDict

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score

# Random Seed
rng = numpy.random.RandomState(1234)

# mnist_x is a (n_sample, n_feature=784) matrix
mnist = fetch_mldata('MNIST original')
mnist_x, mnist_y = mnist.data.astype("float32")/255.0, mnist.target.astype("int32")

# global flag for begin on testing/training
testing = False

class Layer:
    def __init__(self, in_dim, out_dim, function, use_dropout=False, drop_rate=0.5):
        self.W = theano.shared(rng.uniform(low=-numpy.sqrt(6./(in_dim+out_dim)),
                                           high=numpy.sqrt(6./(in_dim+out_dim)),
                                           size=(in_dim,out_dim)).astype('float32'), name="W")
        self.b = theano.shared(numpy.zeros(out_dim).astype("float32"), name="bias")
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
    
class Dropout():
    def __init__(self, in_dim, p=0.5):
        self.p = p
        srng = theano.tensor.shared_randomstreams.RandomStreams(2345)
        self.mask = srng.binomial(n=1, p=1.-self.p, size=numpy.zeros(in_dim).shape).eval().astype('float32') # eval() is float64??
        self.params = []

train_x, valid_x, train_y, valid_y = train_test_split(mnist_x, mnist_y, test_size=0.2, random_state=42)
train_x, test_X, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=33)

x, t = T.fmatrix("x"), T.ivector("t")
def softsign(x):
    return x / (1 + abs(x))
def ReLU(x):
    return T.maximum(0.0,x)

## Settings
# activation = T.nnet.sigmoid #T.tanh, softsign, ReLU
activation = ReLU
learning_rate = 0.01
momentum_rate = 0.9
dropout_rate = 0.5
layers = [
    Layer(784, 500, activation),
    Layer(500, 500, activation, use_dropout=True, drop_rate=dropout_rate),
    Layer(500, 500, activation, use_dropout=True, drop_rate=dropout_rate),
    Layer(500, 10, T.nnet.softmax)
]

## Collect Parameters and Symbolic output
params = []
for i, layer in enumerate(layers):
    params += layer.params
    if i == 0:
        layer_out = layer.fprop(x)
    else:
        layer_out = layer.fprop(layer_out)

y = layers[-1].h
        
## Cost Function (Negative Log Likelihood)

def NLL(): # Negative Log Likelihood
    ## not sum but mean: http://deeplearning.net/tutorial/logreg.html
    return - T.mean(T.log(y)[T.arange(t.shape[0]),t])
def MSE(): # Mean Squared Error
    return T.mean((1-y)[T.arange(t.shape[0]),t] ** 2)
def CrsEnt(): # Binary Cross Entropy
    return T.mean(T.nnet.binary_crossentropy(y,T.extra_ops.to_one_hot(t,10)))

# cost = {NLL, MSE, CrsEnt}
cost = CrsEnt()

# Get Gradient
gparams = T.grad(cost, params)

# Learning Rule
lr = theano.shared(numpy.float32(learning_rate))
momentum = numpy.float32(momentum_rate)

# SGD
def SGD():
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        updates[param] = param - lr * gparam
    return updates

# SGD with momentum
def SGD_with_momentum():
    updates = OrderedDict()
    pdiffs = [theano.shared(numpy.zeros(p.shape.eval()).astype('float32')) for p in params]
    for param, gparam, param_diff in zip(params, gparams, pdiffs):
        pd = pram_diff * momentum - lr * gparam
        updates[param] = param + pd
        updates[param_diff] = pd
    return updates

# AdaGrad
def AdaGrad():
    updates = OrderedDict()
    sumgrads = [theano.shared(numpy.zeros(p.shape.eval()).astype('float32')) for p in params]
    for param, gparam, sumgrad in zip(params, gparams, sumgrads):
        sgrad = sumgrad  + gparam * gparam
        param_diff = - (lr / T.sqrt(sgrad + numpy.float32(1.e-6))) * gparam
        updates[param] = param + param_diff
        updates[sumgrad] = sgrad
    return updates
    
## Compile
# updates = {SGD, SGD_with_momentum, AdaGrad}
train = theano.function([x,t], cost, updates=AdaGrad())
valid = theano.function([x,t], [cost, T.argmax(y, axis=1)])
test = theano.function([x], T.argmax(y, axis=1))

def testSetting(flag):
    testing = flag

## Iterate
batch_size = 100
nbatches = train_x.shape[0]
for epoch in range(100):
    train_x, train_y = shuffle(train_x, train_y)
    for i in range(nbatches):
        start = i * batch_size
        end = start + batch_size

        train(train_x[start:end], train_y[start:end])
        if i % 10000 ==0: print 'training: batch=%s' % i
    testSetting(True)
    valid_cost, pred_y = valid(valid_x, valid_y)
    f1 = f1_score(valid_y, pred_y, average='micro')
    print 'epoch::%s valid_cost::%s f1::%s' % (epoch, valid_cost, f1)
    pred_y = test(test_X)
    print 'epoch::%s test_f1::%s' % (epoch, f1_score(test_y, pred_y, average='micro'))
    testSetting(False)


