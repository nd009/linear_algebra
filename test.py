import unittest
import numpy as np

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
                self.assertTrue(loss<0.1,"Bad result.")

if __name__ == '__main__':
    unittest.main()
