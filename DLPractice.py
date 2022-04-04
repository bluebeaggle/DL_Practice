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



