from test import LinearRegressionTestCase
import unittest
import numpy as np
import copy

def test_matxMultiply():

    for _ in range(2):
        r,d,c = np.random.randint(low=1,high=7,size=3)
        mat1 = np.random.randint(low=-10,high=10,size=(r,d))
        mat2 = np.random.randint(low=-5,high=5,size=(d,c))
        dotProduct = np.dot(mat1,mat2)

        dp = np.array(matxMultiply(mat1.tolist(),mat2.tolist()))

        if not (dotProduct == dp).all():
            print('m:')
            print(mat1)
            print(mat2)

            print('dp:')
            print(dotProduct)
            print(dp)
            print('============')


def matxMultiply(A, B):
    if len(A[0]) != len(B):
        return None
    result_row = len(A)
    result_col = len(B[0])
    result = [[None]*result_col]*result_row
    for i in range(result_row):
        for j in range(result_col):
            result[i][j] = sum([A[i][k]*B[k][j] for k in range(len(B))])


    return result


test_matxMultiply()
