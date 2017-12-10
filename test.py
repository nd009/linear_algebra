# encoding: utf-8
import unittest
import numpy as np
import copy

from decimal import *


class LinearRegressionTestCase(unittest.TestCase):
    """Test for linear regression project"""

    def test_shape(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.randint(low=-10,high=10,size=(r,c))
            self.assertEqual(shape(matrix.tolist()),(r,c),'Wrong answer')


    def test_matxRound(self):

        for decpts in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            dec_true = [[Decimal(str(round(num,decpts))) for num in row] for row in mat]

            matxRound(mat,decpts)
            dec_test = [[Decimal(str(num)) for num in row] for row in mat]

            res = Decimal('0')
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    res += dec_test[i][j].compare_total(dec_true[i][j])

            self.assertEqual(res,Decimal('0'),'Wrong answer')

    def test_transpose(self):
        for _ in range(100):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            t = np.array(transpose(mat))

            self.assertEqual(t.shape,(c,r),"Expected shape{}, but got shape{}".format((c,r),t.shape))
            self.assertTrue((matrix.T == t).all(),'Wrong answer')

    def test_matxMultiply(self):

        for _ in range(100):
            r,d,c = np.random.randint(low=1,high=25,size=3)
            mat1 = np.random.randint(low=-10,high=10,size=(r,d)) 
            mat2 = np.random.randint(low=-5,high=5,size=(d,c)) 
            dotProduct = np.dot(mat1,mat2)

            dp = np.array(matxMultiply(mat1.tolist(),mat2.tolist()))

            self.assertTrue((dotProduct == dp).all(),'Wrong answer')

        mat1 = np.random.randint(low=-10,high=10,size=(r,5)) 
        mat2 = np.random.randint(low=-5,high=5,size=(4,c)) 
        with self.assertRaises(ValueError,msg="Matrix A\'s column number doesn\'t equal to Matrix b\'s row number"):
            matxMultiply(mat1.tolist(),mat2.tolist())

    def test_augmentMatrix(self):

        for _ in range(50):
            r,c = np.random.randint(low=1,high=25,size=2)
            A = np.random.randint(low=-10,high=10,size=(r,c))
            b = np.random.randint(low=-10,high=10,size=(r,1))
            Amat = A.tolist()
            bmat = b.tolist()

            Ab = np.array(augmentMatrix(Amat,bmat))
            ab = np.hstack((A,b))

            self.assertTrue(A.tolist() == Amat,"Matrix A shouldn't be modified")
            self.assertTrue((Ab == ab).all(),'Wrong answer')

    def test_swapRows(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1, r2 = np.random.randint(0,r, size = 2)
            swapRows(mat,r1,r2)

            matrix[[r1,r2]] = matrix[[r2,r1]]

            self.assertTrue((matrix == np.array(mat)).all(),'Wrong answer')

    def test_scaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            rr = np.random.randint(0,r)
            with self.assertRaises(ValueError):
                scaleRow(mat,rr,0)

            scale = np.random.randint(low=1,high=10)
            scaleRow(mat,rr,scale)
            matrix[rr] *= scale

            self.assertTrue((matrix == np.array(mat)).all(),'Wrong answer')
    
    def test_addScaledRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1,r2 = np.random.randint(0,r,size=2)

            scale = np.random.randint(low=1,high=10)
            addScaledRow(mat,r1,r2,scale)
            matrix[r1] += scale * matrix[r2]

            self.assertTrue((matrix == np.array(mat)).all(),'Wrong answer')


    def test_gj_Solve(self):

        for _ in range(9999):
            r = np.random.randint(low=3,high=10)
            A = np.random.randint(low=-10,high=10,size=(r,r))
            b = np.arange(r).reshape((r,1))

            x = gj_Solve(A.tolist(),b.tolist(),epsilon=1.0e-8)

            if np.linalg.matrix_rank(A) < r:
                self.assertEqual(x,None,"Matrix A is singular")
            else:
                self.assertNotEqual(x,None,"Matrix A is not singular")
                self.assertEqual(np.array(x).shape,(r,1),"Expected shape({},1), but got shape{}".format(r,np.array(x).shape))
                Ax = np.dot(A,np.array(x))
                loss = np.mean((Ax - b)**2)
                self.assertTrue(loss<0.1,"Bad result. \nmatrix={}, \nx={}\nb={},\nloss={}".format(A, x, b, loss))


def gj_Solve(A, b, decPts=6, epsilon=1.0e-16):
    # print("matrix A:\n", A, "\n b=\n", b)
    rlen1, clen1 = shape(A)
    rlen2, clen2 = shape(b)

    if clen1 != rlen2:
        return None
    matrix = augmentMatrix(A, b)
    mrlen, mclen = shape(matrix)

    for r in range(mrlen):
        for r_t in range(r, mrlen):
            if matrix[r_t][r] != 0:
                if r_t != r:
                    swapRows(matrix, r, r_t)
                break
        if matrix[r][r] == 0:
            return None
        p_a = matrix[r][r]
        scaleRow(matrix, r, 1.0 / p_a)
        if r == mrlen - 1:
            break
        for r2 in range(r + 1, mrlen):
            if matrix[r2][r] == 0:
                continue
            p_ar = matrix[r2][r] * -1
            addScaledRow(matrix, r2, r, p_ar, epsilon)

    print("gauss matrix={}".format(matrix))
    c_count = mclen - 2
    for rj in range((mrlen - 1), 0, -1):
        for r_2 in range((rj - 1), -1, -1):
            tmp_r_2 = matrix[r_2][c_count] * -1
            if tmp_r_2 == 0:
                continue
            addScaledRow(matrix, r_2, rj, tmp_r_2, epsilon)
        c_count -= 1

    x_v = [[0 for col in range(clen2)] for row in range(rlen1)]
    for r_x in range(rlen1):
        for c_x in range(clen2):
            x_v[r_x][c_x] = round(matrix[r_x][mclen - 1], decPts)
    return x_v


def addScaledRow(M, r1, r2, scale, epsilon=1.0e-16):
    m2 = copy.deepcopy(M)
    scaleRow(m2, r2, scale)
    # print("m2 is \n", m2)
    rlen, clen = shape(M)
    for i in range(clen):
        M[r1][i] = M[r1][i] + m2[r2][i]
        if M[r1][i] > 0 and M[r1][i] < epsilon:
            M[r1][i] = 0.0


def scaleRow(M, r, scale):
    rlen, clen = shape(M)
    if scale == 0:
        raise ValueError("scale cannot be zero")
    for i in range(clen):
        M[r][i] = M[r][i] * scale


def swapRows(M, r1, r2):
    rlen, clen = shape(M)
    for i in range(clen):
        v_r1 = M[r1][i]
        v_r2 = M[r2][i]
        M[r1][i] = v_r2
        M[r2][i] = v_r1


def augmentMatrix(mat1, mat2):
    rlen1, clen1 = shape(mat1)
    rlen2, clen2 = shape(mat2)
    clen3 = clen1 + clen2
    matrix = [[0 for col in range(clen3)] for row in range(rlen1)]
    for i in range(rlen1):
        for j in range(clen3):
            if j < clen1:
                matrix[i][j] = mat1[i][j]
            if j >= clen1:
                k = j - clen1
                matrix[i][j] = mat2[i][k]
    return matrix


def matxMultiply(mat1, mat2):
    rlen1, clen1 = shape(mat1)
    rlen2, clen2 = shape(mat2)
    if clen1 != rlen2:
        raise ValueError("matrix1的列数不等于matrix2的行数，不能进行乘运算")
    matrix3 = [[]] * rlen1
    for i in range(rlen1):
        matrix3[i] = []
        for j in range(clen2):
            v_sum = 0
            for k in range(clen1):
                v_sum += mat1[i][k] * mat2[k][j]
            matrix3[i].append(v_sum)
    return matrix3


def transpose(mat):
    rlen, clen = shape(mat)
    matrix = [[]] * clen

    for i in range(clen):
        matrix[i] = []
        for j in range(rlen):
            matrix[i].append(mat[j][i])
    return matrix


def shape(M):
    return len(M), len(M[0])


def matxRound(mat, decpts):
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            mat[i][j] = round(mat[i][j], decpts)


if __name__ == '__main__':
    unittest.main()
