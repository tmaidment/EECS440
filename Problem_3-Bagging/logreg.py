import mldata
import numpy as np
import utils
import argparse
import os
PRNG = 12345

class LogisticRegression(object):

    def __init__(self, exampleset, constant, weights, epochs=30, lr=0.0001):
        self.dataset = exampleset #utils._convert_exampleset_to_dataframe(exampleset) # convert exampleset to dataframe
        self.constant = constant

        self.data = self.dataset.iloc[:,1:-1].values
        self.data /= self.data.max() # normalize between 0 - 1 to prevent overflow errors

        self.labels = self.dataset.iloc[:,-1].values

        #self.w = 0.002 * np.random.rand(self.data.shape[1]) - 0.001 # random weights between [-0.001, 0.001]
        #self.b = 0.002 * np.random.rand(1) - 0.001
        self.w = np.zeros(self.data.shape[1])
        self.b = 0

        #self.weights = np.ones(self.data.shape[0])
        self.weights = weights

        self.pos = self.dataset.loc[self.dataset.iloc[:,-1] == 1.0] # positive classes
        self.neg = self.dataset.loc[self.dataset.iloc[:,-1] == 0.0] # negative classes

        self.epochs = epochs
        self.lr = lr

        self.w = self.gradient_descent(epochs, lr, self.w, self.b)

    def conditional_log_likelihood_pos(self, data): # calculate the log conditional likelihood for positive class
        cll_mat = np.empty(data.shape[0]) # init matrix to for each data point
        for index in range(data.shape[0]):
            row = data.iloc[index]
            cll_mat[index] = np.log(self.sigmoid(np.dot(self.w, row.iloc[1:-1].values) + self.b)) # add log conditional likelihood to matrix
        return sum(cll_mat) #sum it up

    def conditional_log_likelihood_neg(self, data): # calculate the log condition likelihood for negative class
        cll_mat = np.empty(data.shape[0])
        for index in range(data.shape[0]):
            row = data.iloc[index]
            cll_mat[index] = np.log(1 - self.sigmoid(np.dot(self.w, row.iloc[1:-1].values) + self.b))
        return sum(cll_mat)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient_descent(self, epochs, lr, w, b):
        for i in range(epochs):
            predict = self.sigmoid(np.dot(w, self.data.T) + b)# logistic regression equation
            error = predict - self.labels # subtract to determine error
            #error *= self.weights 
            error = np.multiply(error, self.weights)
            gradient = np.dot(self.data.T, error) + self.constant * np.sum(w) # gradient of w, wit respect to error and gradient of weight decay term
            # multiply it element-wise times the gradient
            w -= lr * gradient # note that gradient is subtracted, to minimize error.  learning rate prevents crazy gradient size.
            #print("iter:", i, "log-conditional likelihood (+ weight decay):", -(self.conditional_log_likelihood_pos(self.pos) + self.conditional_log_likelihood_neg(self.neg) + self.constant * 1/2 * np.sum(np.power(w, 2))))
        print("log-conditional likelihood (+ weight decay):", -(self.conditional_log_likelihood_pos(self.pos) + self.conditional_log_likelihood_neg(self.neg) + self.constant * 1/2 * np.sum(np.power(w, 2))))
        return w

    def predict(self, data):
        val = data.values
        pred = np.zeros([data.shape[0],2])
        for index in range(data.shape[0]):
            if (np.dot(self.w, val[index,1:-1]) + self.b) > 0:
                pred[index][1] = 1
            else:
                pred[index][1] = 0
        #print(pred)
        return pred.astype(int)

    def ensemblePrediction(self, data):
        prediction = self.predict(data)
        return prediction[:,1]

if __name__ == '__main__':
    path = '../voting'
    data = utils._convert_exampleset_to_dataframe(mldata.parse_c45(path.split('/')[-1], path))
    logreg = LogisticRegression(data, constant=0, weights = None)
    print('Final Weights', logreg.w)
    '''
    parser = argparse.ArgumentParser(description='Logistic Regression Implementation')
    parser.add_argument('options', nargs=3, help="The options as specified by the prompt.")
    args = parser.parse_args()

    path = str(args.options[0])
    if(os.path.isdir(path)):
        file_base = next(el for el in reversed(path.split('/')) if el)
        exampleSet = mldata.parse_c45(file_base, path)
        schema = exampleSet.schema
        print("Loading dataset:", file_base)
    else:
        assert 'Dataset input not found!'

    xval = int(args.options[1])
    if(xval == 0):
        print("Cross Validation enabled")
    elif(xval == 1):
        print("Cross Validation disabled")
    else:
        assert 'Unable to determine cross validation flag.'

    constant = int(args.options[2])
    if(constant >= 0):
        print("Constant set to", constant)
    else:
        assert 'Negative Bins is not supported.'

    #logreg(schema, exampleSet, validationType=xval, constant=constant)
    '''