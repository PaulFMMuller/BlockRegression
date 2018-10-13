'''
Written and developed by Paul Muller.
'''

from sklearn.tree import DecisionTreeRegressor
import numpy as np


class BlockGradientBoostingRegressor:
    def __init__(self, n_trees=10, learning_rate=0.1, warm_start=False, **kwargs):
        self.trees = []
        self.n_trees = n_trees
        self.treeArgs = kwargs
        self.warm_start = warm_start
        if learning_rate is None:
            self.gammas = []
            self.search_gamma = True
        else:
            self.search_gamma = False
            self.learning_rate = learning_rate


    def predict(self, X):
        result = np.zeros(X.shape[0])
        for i in range(len(self.trees)):
            if self.search_gamma:
                gamma_i = self.gammas[i]
            else:
                gamma_i = self.learning_rate
            result += gamma_i * self.trees[i].predict(X)
        return result


    def getTreeTargetsPerBlock(self, X, y):
        yPred = self.predict(X).reshape(-1)
        lossGradient = (y - np.sum(yPred)) / X.shape[0]
        target = np.repeat(lossGradient, X.shape[0])
        return X,target


    def getTreeTargets(self, Xs, ys):
        X = np.array([])
        y = np.array([])
        for i in range(len(Xs)):
            Xc = Xs[i]
            if Xc.shape[0] > 0:
                yc = ys[i]
                Xn,yn = self.getTreeTargetsPerBlock(Xc, yc)
                if X.shape[0] == 0:
                    X = Xn
                    y = yn
                else:
                    X = np.vstack([X,Xn])
                    y = np.concatenate((y,yn))
        return X,y


    '''
    Xs is a list of numpy arrays, yc is a list of target sum values.
    The sum of each "real" target variable of the elements of Xs[i] is equal to ys[i].
    '''
    def fit(self, Xs, ys):
        if not self.warm_start:
            self.trees = []
            self.gammas = []

        for i in range(self.n_trees):
            X,y = self.getTreeTargets(Xs, ys)
            newTree = DecisionTreeRegressor(**self.treeArgs)
            newTree.fit(X, y)
            self.trees.append(newTree)
            if self.search_gamma:
                self.gamma_search(X, y)

        return self


    def partial_predict(self, X):
        result_old = np.zeros(X.shape[0])
        for i in range(len(self.trees)-1):
            if self.search_gamma:
                gamma_i = self.gammas[i]
            else:
                gamma_i = self.learning_rate
            result_old += gamma_i * self.trees[i].predict(X)
        result_new = self.trees[-1].predict(X)
        return result_old, result_new


    def getGammaComponents(self, Xs):
        M = np.zeros(len(Xs))
        N = np.zeros(len(Xs))
        for i in range(len(Xs)):
            Xc = Xs[i]
            result_old,result_new = self.partial_predict(Xc)
            M[i] = np.sum(result_new.reshape(-1))
            N[i] = np.sum(result_old.reshape(-1))
        return M,N


    def gamma_search(self, Xs, ys):
        M,N = self.getGammaComponents(Xs)
        delta = np.sum(np.array(ys) - N*M)
        Msq = np.sum(M**2)
        if Msq < 1e-8:                      # In case the problem is undetermined, keep standard value
            self.gammas.append(1)
        else:
            self.gammas.append(delta / Msq)
































