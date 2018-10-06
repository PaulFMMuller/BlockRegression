# BlockRegression

Block Regression a library developed through needs arisen through real problems.
The regression problem it solves is the following.

Given a series of observations X, we suppose there exists a set of corresponding labels y. Nevertheless, we do not have access to these y for training. What we do have access to, is a set of subsets of Xs of X (Not necessarily with null intersection), and the sum Ys of the y corresponding to the X in Xs.

These conditions seem detrimental to the doing of Machine Learning. They are nevertheless solvable, and we implemented a Linear Regression algorithm, and a Gradient Boosting algorithm to solve these problems.

Usage is straightforward and conforms to the notations defined above.
