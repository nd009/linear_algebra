import numpy as np 

def generateMatrix(rank=4,seed=None,singular=False):
    np.random.seed(seed)
    while True:
        matrix = np.random.randint(-10,10, size=(rank, rank))
        if (np.linalg.matrix_rank(matrix) != rank) ^ (not singular):
            return matrix

def printInMatrixFormat(rank,A,b):
    rowFormat = ','.join(["{:>5.0f}"] * rank) + " || {:<5.0f}"     
    matrixFormat = '\n'.join([rowFormat] * rank)
    
    Ab = lambda A,b: [ra+rb for ra,rb in zip(A,b)]
    flattern = [e for row in Ab(A.tolist(),b.tolist()) for e in row]

    print(matrixFormat.format(*flattern))

def generatePoints(seed=None,num=100):
    np.random.seed(seed)
    m = np.random.random() * 10 - 5 # -5 ~ 5
    b = np.random.random() * 10 + 5 # 5 ~ 15

    x = np.random.random(size=num) * 10 - 5
    y = x * m + b 
    y += np.random.normal(size=num)

    return x,y

