#!/usr/bin/env python3
# coding:utf-8
import time, struct
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


def load_mnist():
    raw_train_img = 'train-images.idx3-ubyte'
    raw_train_lbl = 'train-labels.idx1-ubyte'
    raw_test_img = 't10k-images.idx3-ubyte'
    raw_test_lbl = 't10k-labels.idx1-ubyte'

    with open(raw_train_img,'rb') as f:
        f.seek(16)  # move cursor
        train_img = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_img = train_img.reshape(60000, 784).astype('float32')

    with open(raw_train_lbl,'rb') as f:
        f.seek(8)  # move cursor
        train_lbl = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))

    with open(raw_test_img,'rb') as f:
        f.seek(16)  # move cursor
        test_img = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_img = test_img.reshape(10000, 784).astype('float32')

    with open(raw_test_lbl,'rb') as f:
        f.seek(8)  # move cursor
        test_lbl = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))

    return (train_img, train_lbl), (test_img, test_lbl)

def one_hot_encoding(data, rows):
    idx = 0
    result = np.zeros((rows, 10), dtype=data.dtype)
    for val in data:
        result[idx, val] = 1
        idx += 1
    
    return result

def sigmoid(input):
    return 1 / (1 + np.exp(-input))

def softmax(input):
    max = np.max(input)
    exp_input = np.exp(input - max)  # prevent overflow
    exp_sum = np.sum(exp_input)

    return exp_input / exp_sum

def cross_entropy_loss(predict, label):
    delta = 1e-7
    if predict.ndim == 1:  # e.g. np.shape(4,) => np.shape(1, 4)
        predict = predict.reshape(1, predict.size)
        label = label.reshape(1, label.size)

    return -np.sum(label * np.log(predict + delta)) / predict.shape[0]

def numerical_gradient(func, params):
    h = 1e-4
    grad = np.zeros_like(params)

    it = np.nditer(params, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp = params[idx]
        params[idx] = tmp + h
        forward_diff = func()
        
        params[idx] = tmp - h 
        backward_diff = func()

        grad[idx] = (forward_diff - backward_diff) / (2 * h)       
        params[idx] = tmp 
        it.iternext()

    return grad

class MnistNeuralNet:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, weight_init=0.01):
        self.params = {}
        self.params['input_weight'] = weight_init * np.random.randn(input_neurons, hidden_neurons)
        self.params['input_bias'] = np.zeros(hidden_neurons)
        self.params['hidden_weight'] = weight_init * np.random.randn(hidden_neurons, output_neurons)
        self.params['hidden_bias'] = np.zeros(output_neurons)
    
    def predict(self, img):
        input_w, hidden_w = self.params['input_weight'], self.params['hidden_weight']
        input_b, hidden_b = self.params['input_bias'], self.params['hidden_bias']

        input_layer = np.dot(img, input_w) + input_b
        input_layer = sigmoid(input_layer)

        hidden_layer = np.dot(input_layer, hidden_w) + hidden_b
        hidden_layer = softmax(hidden_layer)

        return hidden_layer

    def loss(self, img, label):
        predict = self.predict(img)

        return cross_entropy_loss(predict, label) 

    def accuracy(self, img, label):
        predict = np.argmax(self.predict(img), axis=1)
        label = np.argmax(label, axis=1)
        acc = np.sum(predict == label) / float(predict.shape[0])

        return acc
    
    def gradient(self, img, label):
        loss = lambda : self.loss(img, label)
        
        grads = {}
        grads['input_weight'] = numerical_gradient(loss, self.params['input_weight'])
        grads['input_bias'] = numerical_gradient(loss, self.params['input_bias'])
        grads['hidden_weight'] = numerical_gradient(loss, self.params['hidden_weight'])
        grads['hidden_bias'] = numerical_gradient(loss, self.params['hidden_bias'])

        return grads

# load mnist data      
(train_img, train_lbl), (test_img, test_lbl) = load_mnist()

# data preprocessing
train_img = train_img / 255
test_img = test_img / 255
train_lbl = one_hot_encoding(train_lbl, 60000)
test_lbl = one_hot_encoding(test_lbl, 10000)

# training setup
lr = 0.1  # learning rate
epoch = 10
batch_size = 100
train_size = train_img.shape[0]
epoch_size = train_size / batch_size
iterations = epoch * epoch_size

train_lost = []
train_accuracy = []
test_accuracy = []
net = MnistNeuralNet(784, 50, 10)

# batch processing
def batch_processing(i):
    batch_idx = np.random.choice(train_size, batch_size)
    batch_img = train_img[batch_idx]
    batch_lbl = train_lbl[batch_idx]

    grad = net.gradient(batch_img, batch_lbl)

    for key in ('input_weight', 'input_bias', 'hidden_weight', 'hidden_bias'):
        net.params[key] -= lr * grad[key]

    if i % epoch_size == 0:
        loss = net.loss(batch_img, batch_lbl)
        train_acc = net.accuracy(train_img, train_lbl)
        test_acc = net.accuracy(test_img, test_lbl)

        train_lost.append(loss)
        train_accuracy.append(train_accuracy)
        test_accuracy.append(test_acc)
        print('epoch ' + str(i / epoch_size) + ' | train lost: ' + \
            str(loss) + ' | train acc: '+ str(train_acc) + ' | test acc: ' + str(test_acc))
    
    return 0

time_start = time.time()
pool = mp.Pool()
res = pool.map(batch_processing, range(1, int(iterations + 1)))
time_end = time.time()
print('training cost : %.2f sec' % (time_end - time_start))