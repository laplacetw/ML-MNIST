#!/usr/bin/env python3
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(10)


def showImg(img):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(img, cmap='binary')
    plt.show()


def showPredict(imgs, lbls, predicts, startID, num=10):
    plt.gcf().set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(imgs[startID], cmap='binary')

        if len(predicts) > 0:
            title = 'ai = ' + str(predicts[i])
            if predicts[i] == lbls[i]:
                title += '(O)'
            else:
                title += '(X)'

            title += '\nlabel = ' + str(lbls[i])
        else:
            title += 'label = ' + str(lbls[i])

        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        startID += 1
    
    plt.show()


# load mnist data
(train_feature, train_label), (test_feature, test_label) = mnist.load_data()

# show the dimension of data : (60000, 28, 28)
#print(train_feature.shape)

# data preprocessing
# reshape
train_feature_vector = train_feature.reshape(len(train_feature), 784).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature), 784).astype('float32')

# feature normalization 
train_feature_normal = train_feature_vector / 255
test_feature_normal = test_feature_vector / 255

# one-hot encoding
train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)

# model definition
model = Sequential()

# input:784, hidden:256, output:10
model.add( Dense(units=256, input_dim=784, kernal_initializer='normal', activation='relu') )
model.add( Dense(units=10, kernal_initializer='normal', activation='softmax') )

# training definition
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=train_feature_normal, y=train_label_onehot,
                          validation_split=0.2, epochs=10, batch_size=200, verbose=2)

# accuracy evaluation
accuracy = model.evaluate(test_feature_normal, test_label_onehot)
print('\n[Accuracy] = ', accuracy[1])

# prediction
prediction = model.predict_classes(test_feature_normal)
showPredict(test_feature, test_label, prediction, 0)