import sys; sys.path.append('../Library') ; from BlockGradientBoostingRegressor import BlockGradientBoostingRegressor
from TestFunction import generateDataset, randomlyGenerateBlocks
import numpy as np


np.random.seed(0)


N = 10000
M = 5
P = 150
sigma_noise = 0.05


X,y,coef_y = generateDataset(N,M,sigma_noise=sigma_noise)
Xs,ys = randomlyGenerateBlocks(X,y,P)

Regressor = BlockGradientBoostingRegressor().fit(Xs,ys)


yPred = Regressor.predict(X)
print(np.mean(np.abs(y-yPred)))