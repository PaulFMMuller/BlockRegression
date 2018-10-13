import matplotlib.pyplot as plt
import numpy as np



def defaultMuBehaviour(mu, M):
    if mu is None:
        mu = np.zeros((1, M))
    return mu


def defaultSigmaBehaviour(sigma, M):
    if sigma is None:
        sigma = np.ones((1, M))


def generateDataset(N, M, mu_x=None, mu_y=None, sigma_x=None, sigma_y=None):
    mu_x = defaultMuBehaviour(mu_x, M)
    mu_y = defaultMuBehaviour(mu_y, M)

    sigma_x = defaultSigmaBehaviour(sigma_x, M)
    sigma_y = defaultSigmaBehaviour(sigma_y, M)

    X = mu_x + sigma_x * np.random.normal(size=(N, M))
    coef_y = (mu_y + sigma_y * np.random.normal(size=(1, M))).reshape(-1,1)
    y = X.dot(coef_y)

    return X,y,coef_y


def randomlyGenerateBlocks(P,X,y):
    N = X.shape[0]
    if P > N:
        raise Exception('Too many splits required. Impossible.')
    nSplits = 0
    while





























