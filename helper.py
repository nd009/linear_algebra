import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

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


def generatePoints2D(seed):
    np.random.seed(seed)

    num = np.random.randint(128,256)
    m = np.random.random() * 10 - 5 # -5 ~ 5
    b = np.random.random() * 10 + 5 # 5 ~ 15

    x = np.random.random(size=num) * 10 - 5
    y = x * m + b
    y += np.random.normal(size=num)

    return x.tolist(),y.tolist()

def generatePoints3D(seed):
    np.random.seed(seed)

    num = np.random.randint(128,256)
    X = np.linspace(-5, 5, num)
    m = np.random.random(2) * 10 - 5
    b = np.random.random() * 15 + 5
    X = np.vstack([X]*2).T + np.random.randn(num, 2)
    Y = np.dot(X, m) + b + np.random.normal(size=num)
    return X.tolist(), Y.tolist()

def vs_scatter_2d(X, Y, m=None, b=None):
    plt.figure()
    x_vals = (-5, 5)
    plt.xlim(x_vals)
    plt.xlabel('x',fontsize=18)
    plt.ylabel('y',fontsize=18)
    plt.scatter(X,Y,c='b')

    if  m != None and b != None:
        y_vals = [m*x+b for x in x_vals]
        plt.plot(x_vals, y_vals, '-', color='r')

    plt.show()



def vs_scatter_3d(X, Y, coeff=None):
    title = 'target points'
    cmap = cm.get_cmap('gist_rainbow')
    fig = plt.figure()
    axe3d = fig.gca(projection = '3d')
    axe3d.scatter(list(zip(*X))[0], list(zip(*X))[1], Y, linewidth = 0)

    if coeff:
        title = 'linear regression on target points'
        x = [-5 , 5]
        y = [-5,  5]
        z = np.dot(np.transpose([x, y]), coeff[:2]) + coeff[2]
        axe3d.plot(x, y, z, c='r')
    plt.show()
