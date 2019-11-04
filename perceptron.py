#!/usr/bin/env python3
# coding:utf-8
import numpy as np
import matplotlib.pylab as plt

def OR(x1, x2):
    signal = np.array([x1, x2])
    weight = np.array([0.5, 0.5])
    bias = -0.3
    activation = np.sum(signal * weight) + bias
    output = 1 if activation > 0 else 0
    return output

def AND(x1, x2):
    signal = np.array([x1, x2])
    weight = np.array([0.5, 0.5])
    bias = -0.7
    activation = np.sum(signal * weight) + bias
    output = 1 if activation > 0 else 0
    return output

def NAND(x1, x2):
    signal = np.array([x1, x2])
    weight = np.array([-0.5, -0.5])
    bias = 0.7
    activation = np.sum(signal * weight) + bias
    output = 1 if activation > 0 else 0
    return output

def XOR(x1, x2):
    activation = AND( OR(x1, x2), NAND(x1, x2) )
    output = 1 if activation > 0 else 0
    return output

def logic_gate_test(name, gate):
    print(name, ' :')
    print(0, 0, gate(0, 0))
    print(0, 1, gate(0, 1))
    print(1, 0, gate(1, 0))
    print(1, 1, gate(1, 1))

def step_func(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

logic_gate_test('OR', OR)
logic_gate_test('AND', AND)
logic_gate_test('NAND', NAND)
logic_gate_test('XOR', XOR)

# line plot
x = np.arange(-5.0, 5.0, 0.1)
y1 = step_func(x)
y2 = sigmoid(x)
plt.plot(x, y1, label="step_func")
plt.plot(x, y2, label="sigmoid", linestyle="--")
plt.legend()
plt.show()