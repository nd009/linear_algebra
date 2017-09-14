import numpy as np 
import matplotlib.pyplot as plt

def generateSingularMatrix(rank=4):
	while True:
		matrix = np.random.randint(-10,10,size=(rank,rank))
		if np.linalg.matrix_rank(matrix) < rank:
			return matrix


def generatePoints(num=100, col=2, outliner=0):
    m,b = np.random.randint(-5,5,size=col).tolist()

    x = np.random.random(size=num) * 10 - 5
    y = x * m + b 
    y += np.random.normal(size=num)

    return x,y

