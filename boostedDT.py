'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
from sklearn import tree
import pdb

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
        self.numBoostingIters = numBoostingIters
        self.maxTreeDepth = maxTreeDepth
    

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''

        self.findNumClasses(y)

        n, d = X.shape
       
        ratio = 1.0
        ratio = ratio/n
        weights = [ratio] * n

        self.model = []
        self.betas = []
        self.weights = []

        for t in range(self.numBoostingIters):

            clf = tree.DecisionTreeClassifier(max_depth=self.maxTreeDepth).fit(X, y, sample_weight=weights)

            self.model.append(clf)

            error = 0
            # Calculate errors
            predictions = clf.predict(X)
            for i in range(n):
                if predictions[i] != y[i]:
                    error += weights[i]

            beta = (np.log((1 - error)/error) + np.log(self.num_class - 1))
            self.betas.append(beta)

            # Update weights
            for i in range(n):
                if predictions[i] == y[i]:
                    weights[i] = weights[i] * np.exp(-beta)
                # else:
                #     weights[i] = weights[i]

            total_new_weights = np.sum(weights)

            # Normalize weights
            for i in range(n):
                weights[i] = weights[i]/total_new_weights




    def findNumClasses(self, y):
        max_class = 0
        for i in range(y.size):
            if y[i] > max_class:
                max_class = y[i]

        self.num_class = max_class + 1

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n, d = X.shape

        weighted_sums = np.zeros((n, self.num_class))

        for i in range(self.numBoostingIters):

            iteration_prediction = self.model[i].predict(X)
            iteration_beta = self.betas[i]
            for j in range(n):
                weighted_sums[j, iteration_prediction[j]] += iteration_beta


        result = np.argmax(weighted_sums, axis=1)

        print result

        return result

