import numpy as np
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


