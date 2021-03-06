import sys, os
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image



def img_show(img) :
    pil_img = Image.fromarray(np.unit8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) =\
        load_mnist(flatten=True, normalize=False)


print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28,28)
print(img.shape)

img_show(img)


def get_data():
    (x_train, t_train),(x_test,t_test) =\
            load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = picle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'],network['W2'],network['W3']
    b1, b2, b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)

    return y



x,y = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i] :
        accuracy_cnt += 1

print("Accuracy:"+str(float(accuracy_cnt) / len(x)))

x,_ = get_data()
network = init_network()
W1,W2,W3 = network['W1'],network['W2'],network['W3']

print(x.shape)
print(x[0].shape)
print(W1.shape)
print(W2.shape)
print(W3.shape)


:wq

