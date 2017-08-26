import unittest
import numpy as np

def shape(M):
    return len(M),len(M[0])


def augmentMatrix(A, b):
    return [a+bi for a, bi in zip(A, b)]

def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]

def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError
    for i,ele in enumerate(M[r]):
        M[r][i] = ele*scale

def addScaledRow(M, r1, r2, scale):
    for i,ele in enumerate(M[r1]):
        M[r1][i] = ele + M[r2][i]*scale



def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    height, width = shape(A)
    if height != len(b):
        return None
    Ab = augmentMatrix(A, b)
    for i in range(width):
        # in each column, find the absolute
        j = i  # j is the index of max abs row
        abs_max = abs(Ab[j][i])
        for row in range(i+1, height):
            if abs(Ab[row][i]) > abs_max:
                abs_max = abs(Ab[row][i])
                j = row
        if abs_max < epsilon:
            return None
        swapRows(Ab, i, j)
        scaleRow(Ab, i, scale=1.0/Ab[i][i])
        for row in range(height):
            if row != i:
                addScaledRow(Ab, row, i, -Ab[row][i])
    x = [list(a) for a in zip(*Ab)]
    return [list(a) for a in list(zip(x[width]))]

A = [[-2,8,0,9], [-10,6, -9, -7], [6, -1, 6, 3], [-8, -7, 6, 7]]
b = [[0,], [1, ], [2,], [3, ]]

gj_Solve(A, b)

def test_gj_Solve():

    for _ in range(5):
        r = np.random.randint(low=3,high=5)
        A = np.random.randint(low=-10,high=10,size=(r,r))
        b = np.arange(r).reshape((r,1))

        x = gj_Solve(A.tolist(),b.tolist())

        if np.linalg.matrix_rank(A) < r:
            assertEqual(x, None, "Matrix A is singular")
        else:
            if x is None:
                print("Matrix A is not singular")
            if np.array(x).ndim != 2:
                print("x have to be two-dimensional Python List")
            Ax = np.dot(A,np.array(x))
            loss = np.mean((Ax - b)**2)
            if not loss<0.01:
                print("Regression result isn't good enough")

#test_gj_Solve()
