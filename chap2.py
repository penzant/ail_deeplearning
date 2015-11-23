# coding: utf-8

# ## k−N Nfor MNIST
import numpy
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

X, y = shuffle(mnist.data, mnist.target)
X = X / 255.0
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_X, dev_X , train_y, dev_y = train_test_split(train_X, train_y, test_size=0.2)
# len: X=70000, train=44800, dev=11200, test=14000

# normalization
norm = numpy.linalg.norm(train_X, ord=2, axis=1)
normalized_train_X = train_X / norm[:,numpy.newaxis]
norm = numpy.linalg.norm(dev_X, ord=2, axis=1)
normalized_dev_X = dev_X / norm[:,numpy.newaxis]

# homework
# ### prediction for test data by k-NN 
from collections import defaultdict, Counter
import heapq
ks = defaultdict(lambda: 0)
count = 0
for i in range(len(normalized_dev_X)):
    score = []
    count += 1
    if (count % 100 == 0): print "count:" + str(count)
    for j in normalized_train_X:
        score.append(numpy.dot(normalized_dev_X[i],j))
    # partial sort using heapq: faster than sorted() for this data size
    most = heapq.nlargest(30, zip(score,train_y))
    # counting numbers for each k (= num+1)
    for num in range(len(most)):
        ans = Counter([x[1] for x in most[:num+1]]).most_common()
        if ans[0][0] == dev_y[i]:
            ks[num+1] += 1
print ks.items()
fix_k = max(ks.items(), key=lambda x: x[1])[0]
print "learned k:" + str(fix_k)

pred_y =[]
for i in test_X:
    score = []
    for j in train_X:
        score.append(numpy.dot(i,j))
    most = heapq.nlargest(fix_k, zip(score,train_y))
    ans = Counter([x[1] for x in most]).most_common()
    pred_y.append(ans[0][0])
f1_pred = f1_score(test_y, pred_y, average='micro')
print f1_pred



