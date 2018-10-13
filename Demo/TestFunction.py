import matplotlib.pyplot as plt
import numpy as np



def defaultMuBehaviour(mu, M):
    if mu is None:
        mu = np.zeros((1, M))
    return mu


def defaultSigmaBehaviour(sigma, M):
    if sigma is None:
        sigma = np.ones((1, M))
    return sigma


def generateDataset(N, M, mu_x=None, mu_y=None, sigma_x=None, sigma_y=None, sigma_noise=None):
    mu_x = defaultMuBehaviour(mu_x, M)
    mu_y = defaultMuBehaviour(mu_y, M)

    sigma_x = defaultSigmaBehaviour(sigma_x, M)
    sigma_y = defaultSigmaBehaviour(sigma_y, M)
    sigma_noise = defaultSigmaBehaviour(sigma_noise, 1)

    X = mu_x + sigma_x * np.random.normal(size=(N, M))
    coef_y = (mu_y + sigma_y * np.random.normal(size=(1, M))).reshape(-1,1)
    y = X.dot(coef_y) + sigma_noise * np.random.normal(size = (X.shape[0],1))

    return X,y,coef_y


def randomlyGenerateBlocks(X, y, P):
    N = X.shape[0]
    if P > N:
        raise Exception('Too many splits required. Impossible.')
    indexes = list(range(N-1))
    splitIndexes = [0] + list(np.random.choice(indexes,P)+1)
    indexes = sorted(splitIndexes)
    Xresult = [0] * P
    yResult = [0] * P
    for i in range(P):
        Xresult[i] = X[indexes[i]:indexes[i+1]]
        yResult[i] = np.sum(y[indexes[i]:indexes[i+1]])
    return Xresult,yResult




























