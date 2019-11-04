#!/usr/bin/env python3
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from cv2 import cv2 as cv
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
np.random.seed(1234)  # for reproducibility


def showPredict(imgs, lbls, predictions):
    print("lbls : ", lbls)
    print("prediction : ", prediction)
    plt.gcf().set_size_inches(10, 10)
    for i in range(0, 10):
        fig = plt.subplot(2, 5, i + 1)
        fig.imshow(imgs[i], cmap='binary')

        title = 'prediction = ' + str(predictions[i])
        if predictions[i] != lbls[i]:
            title += '(X)'

        title += '\nlabel = ' + str(lbls[i])
        fig.set_title(title, fontsize=10)
        fig.set_xticks([])
        fig.set_yticks([])
    
    plt.show()

def mdlTrain(train_feature, train_label, test_feature, test_label):
    # model definition
    model = Sequential()

    # input:784, hidden:256, output:10
    model.add( Dense(units=256, input_dim=784, init='normal', activation='relu') )
    model.add( Dense(units=10, init='normal', activation='softmax') )

    # training definition
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x=train_feature, y=train_label, validation_split=0.2, epochs=10, batch_size=200, verbose=2)

    # accuracy evaluation
    accuracy = model.evaluate(test_feature, test_label)
    print('\n[Accuracy] = ', accuracy[1])

    return model

def get_test_process(files):
    test_image = []
    test_label = []
    for file in files:
        label = int(file[0:1])
        image = cv.imread(file, cv.IMREAD_GRAYSCALE)
        image = cv.threshold(image, 120, 255, cv.THRESH_BINARY_INV)[1]  # retval, dst = cv.threshold(src, thresh, maxval, type[,dst])
        test_image.append(image)
        test_label.append(label)

    test_image = np.array(test_image)
    test_label = np.array(test_label)

    test_image_normal = test_image.reshape(len(test_image), 784).astype('float32') / 255
    test_label_onehot = np_utils.to_categorical(test_label)

    return (test_image, test_label), (test_image_normal, test_label_onehot)


'''
# load mnist data
(train_feature, train_label), (test_feature, test_label) = mnist.load_data()

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

action = input("1: Model Testing\n2: Model Training\n")
if action == "1":
    print("Load mdl_mlp_mnist.h5")
    model = load_model("mdl_mlp_mnist.h5")
    prediction = model.predict_classes(test_feature_normal)
    showPredict(test_feature, test_label, prediction)
    del model
else:
    print("===== Start training =====")
    model = mdlTrain(train_feature_normal, train_label_onehot, test_feature_normal, test_label_onehot)
    model.save("mdl_mlp_mnist.h5")
    print("===== Model has been saved =====")
    prediction = model.predict_classes(test_feature_normal)
    showPredict(test_feature, test_label, prediction)
    del model

'''
model = load_model("mdl_mlp_mnist.h5")
print("=== Load mdl_mlp_mnist.h5 ===")
files = glob('*.jpg')
(test_image, test_label), (test_image_normal, test_label_onehot) = get_test_process(files)
prediction = model.predict_classes(test_image_normal)
showPredict(test_image, test_label, prediction)
del model
