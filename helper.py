import numpy as np 

def generateSingularMatrix(rank=4):
    matrix = np.random.randint(-10,10,size=(rank,rank))
    matrix[-1] = np.sum(matrix[:-1], axis =0)
    return matrix

def generateNonSingularMatrix(rank=4):
    while True:
        matrix = np.random.randint(-10,10, size=(rank, rank))
        if (np.linalg.matrix_rank(matrix) == rank):
            return matrix

def generatePoints(num=100):
    m = np.random.random() * 10 - 5 # -5 ~ 5
    b = np.random.random() * 10 + 5 # 5 ~ 15

    x = np.random.random(size=num) * 10 - 5
    y = x * m + b 
    y += np.random.normal(size=num)

    return x,y

