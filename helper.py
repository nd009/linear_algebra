import numpy as np 

def generateMatrix(rank,seed,singular=False):
    np.random.seed(seed)
    while True:
        matrix = np.random.randint(-10,10, size=(rank, rank))
        if (np.linalg.matrix_rank(matrix) != rank) ^ (not singular):
            return matrix

def printInMatrixFormat(Ab,padding=7,truncating=3):
    rank = len(Ab)
    rowFormat = ','.join(["{{:>{}.{}f}}".format(padding,truncating)] * rank) + " || {{:^{}.{}f}}".format(padding,truncating)     
    matrixFormat = '\n'.join([rowFormat] * rank)

    flattern = [e for row in Ab for e in row]

    print(matrixFormat.format(*flattern))


def generatePoints(seed,num=100):
    np.random.seed(seed)
    m = np.random.random() * 10 - 5 # -5 ~ 5
    b = np.random.random() * 10 + 5 # 5 ~ 15

    x = np.random.random(size=num) * 10 - 5
    y = x * m + b 
    y += np.random.normal(size=num)

    return x.tolist(),y.tolist()

