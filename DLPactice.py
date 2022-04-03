import numpy as np

x = np.array([0,1])
w = np.array([0.5,0.5])
b = -7

print(w*x)
print(np.sum(w*x))
print(np.sum(w*x)+b)


def AND(x1,x2) :
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else :
        return 1

def NAND(x1,x2) :
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x) +b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1,x2) :
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x) +b
    if tmp <= 0:
        return 0
    else :
        return 1
print('--------AND---------')
print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))
print('--------NAND--------')
print(NAND(0,0))
print(NAND(0,1))
print(NAND(1,0))
print(NAND(1,1))
print('---------OR---------')
print(OR(0,0))
print(OR(0,1))
print(OR(1,0))
print(OR(1,1))

def XOR(x1,x2) :
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y
print('---------XOR---------')
print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))

#def step_function(x) :
#    if x > 0:
#        return 1
#    else :
#        return 0
def step_function(x) :
    y = x>0
    return y.astype(np.int)
print('-------astype-------')
x = np.array([-1.0,1.0,2.0])
y = x>0
print(y)
y = y.astype(np.int)
print(y)



