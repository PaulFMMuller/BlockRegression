'''
Written and developed by Paul Muller.
'''



import numpy as np


class BlockLinearRegressor:
    def __init__(self, fit_intercept=True, lambda_ridge=None):
        self.beta = []
        if lambda_ridge is None:
            self.ridge = False
        else:
            self.ridge = True
            self.lambda_ridge = lambda_ridge
        self.fit_intercept = fit_intercept

    '''
    Xs is a list of numpy arrays, yc is a list of target sum values.
    The sum of each "real" target variable of the elements of Xs[i] is equal to ys[i].
    '''
    def fit(self, Xs, ys):
        interceptTerm = 0
        if self.fit_intercept:
            interceptTerm  = 1    
        M = np.zeros((Xs[0].shape[1]+interceptTerm, Xs[0].shape[1]+interceptTerm))
        N = np.zeros((Xs[0].shape[1]+interceptTerm,1))
        for i in range(len(Xs)):
            Xc = Xs[i]
            yc = ys[i]

            if self.fit_intercept:
                Xc = np.concatenate([Xc, np.ones((Xc.shape[0], 1))], axis=1)    # Adding intercept column
            ones = np.ones((Xc.shape[0], 1))
            M += np.linalg.multi_dot([np.transpose(Xc), ones, np.transpose(ones), Xc])
            N += yc * np.transpose(Xc).dot(ones)

        if self.ridge:
            M += self.lambda_ridge * np.eyes(M.shape[0])                        # Integrating the Ridge term into the final matrix

        self.beta = np.linalg.solve(M, N)
        return self


    # Standard predict function.
    def predict(self, X):
        if self.fit_intercept:
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return np.matmul(X, self.beta)


    '''
    Predicts the sum of each block contained in Xs.
    Used to check the prediction results against the sum values ys.
    '''
    def predict_block(self, Xs):
        results = [0]*len(Xs)
        for i in range(len(Xs)):
            results[i] = np.sum(self.predict(Xs[i]).reshape(-1))
        return results












