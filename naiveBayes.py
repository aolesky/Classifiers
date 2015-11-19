'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class NaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.useLaplaceSmoothing = useLaplaceSmoothing

    def findNumClasses(self, y):
        max_class = 0
        for i in range(y.size):
            if y[i] > max_class:
                max_class = y[i]

        self.num_class = max_class + 1

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''

        self.findNumClasses(y)

        n, d = X.shape

        self.class_occurences = np.zeros(self.num_class)

        occurences = np.zeros((self.num_class, d))
        total = np.zeros(self.num_class)

        # First find the frequency of each  feature
        for i in range(n):
            self.class_occurences[y[i]] += 1
            for j in range(d):
                occurences[y[i], j] += X[i, j]
                total[y[i]] += X[i, j]

        self.probabilities = np.zeros((self.num_class, d))

        for i in range(self.num_class):
            for j in range(d):
                if self.useLaplaceSmoothing:
                    self.probabilities[i, j] = (occurences[i, j] + 1)/(total[i] + d)
                else:
                    self.probabilities[i, j] = occurences[i, j]/total[i]



    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''

        probabilities = self.predictLogProbs(X)

        result = np.argmax(probabilities, axis=1)

        return result

    def predictLogProbs(self, X):
        n, d = X.shape

        instance_probabilities = np.zeros((n, self.num_class))

        for i in range(n):
            for j in range(self.num_class):
                frequency_sum = 0

                for k in range(d):
                    frequency_sum += (X[i, k] * np.log(self.probabilities[j, k]))

                class_probability = np.log(self.class_occurences[j]/np.sum(self.class_occurences))
                instance_probabilities[i, j] = frequency_sum + class_probability

        return instance_probabilities

    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''

        n, d = X.shape

        log_probs = self.predictLogProbs(X)
        real_probs = np.zeros((n, self.num_class))

        for i in range(n):
            a = float("-inf")
            for j in range(self.num_class):
                if log_probs[i, j] > a:
                    a = log_probs[i, j]

            #normalize
            anti_logged_sum = np.sum(np.exp(log_probs[i, :] - a))
            alpha_log = -(a + np.log(anti_logged_sum))

            for j in range(self.num_class):
                current_log_probability = log_probs[i, j]

                real_probs[i, j] = current_log_probability + alpha_log


        result = np.exp(real_probs)

        return result
        
        
        
class OnlineNaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
    
    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
