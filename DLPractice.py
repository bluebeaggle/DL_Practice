import numpy as np

def sigmoid(x) :
    return 1/(1+np.exp(-x))

A = np.array([1,2,3,4])
print('A는' , A)
print('np.ndim(A)) = ' , np.ndim(A))
print('A.shape = ' , A.shape)
print('A.shape[0] = ' , A.shape[0])

B = np.array([[1,2],[3,4],[5,6]])
print('B = ' , B)
print('np.ndim(B) = ' ,  np.ndim(B))
print('B.shape = ' ,  B.shape)

A = np.array([[1,2],[3,4]])
print('A.shape = ' , A.shape)
B = np.array([[5,6],[7,8]])
print('B.shape = ',B.shape)
print('np.dot(A,B) = ' , np.dot(A,B))

X = np.array([1,2])
print('X.shape = ', X.shape)
W = np.array([[1,3,5],[2,4,6]])
print('W = ', W)
print('W.shape = ', W.shape)
Y = np.dot(X,W)
print('Y = np.dot(X,W) ')
print(Y)

X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

print('W1.shape = ',W1.shape)
print('X.shape = ',X.shape)
print('B1.shape = ', B1.shape)

A1 = np.dot(X,W1) + B1
print('A1 = np.dot(X,W1)+B1')
print(A1)
Z1 = sigmoid(A1)
print('Z1 = sogmoid(A1)')
print(Z1)

W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.3]])
B2 = np.array([0.1,0.2])

print('Z1.shape = ',Z1.shape)
print('W2.shape = ', W2.shape)
print('B2.shape = ',B2.shape)

A2 = np.dot(Z1,W2) +B2
Z2 = sigmoid(A2)

print('A2=np.dot(Z1,W2) +B2')
print(A2)
print('Z2 = sigmoid(A2)')
print(Z2)

def identity_function(x) :
    return x
W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2,W3) + B3
Y = identity_function(A3) #Y = A3

print(Y)




