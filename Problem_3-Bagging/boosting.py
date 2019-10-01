import numpy as np
import utils
import logreg as lr
import nbayes as nb
import decisiontree as dt
import nodeSelector as ns
import mldata
PRNG = 12345

class boosting(object):

    def __init__(self, dataset, algorithm, iterations):
        #self.schema = schema
        #self.dataset = dataset
        self.exampleset = dataset
        self.dataset = utils._convert_exampleset_to_dataframe(dataset)
        self.algorithm = algorithm
        self.iterations = iterations

        # create n weights of 1/n
        self.init_weight = np.ones(self.dataset.shape[0])/self.dataset.shape[0]

        if 'dtree' in self.algorithm :
            self.classifiers, self.classifier_weights = self.adaboost_dtree(self.exampleset, self.iterations)
        elif 'nbayes' in self.algorithm:
            self.classifiers, self.classifier_weights = self.adaboost_nbayes(self.dataset, self.iterations)
        elif 'logreg' in self.algorithm:
            self.classifiers, self.classifier_weights = self.adaboost_logreg(self.dataset, self.iterations)
        else:
            print('Algorithm input is invalid.')

    def predict(self, X):
        if 'dtree' in self.algorithm:
            return self.dtree_predict(X)
        elif 'nbayes' in self.algorithm:
            return self.nbayes_predict(X)
        elif 'logreg' in self.algorithm:
            return self.logreg_predict(X)
        else:
            print('Algorithm input is invalid.')

    def dtree_predict(self, X):
        T = len(self.classifiers)
        Y = utils._convert_exampleset_to_dataframe(X)
        pred = np.zeros([T, Y.shape[0], 2])
        denom = sum(self.classifier_weights)

        for idx, classifier in enumerate(self.classifiers):
            pred[idx] = classifier.predict(X) * self.classifier_weights[idx] / denom

        return np.sum(pred, axis=0)

    def adaboost_dtree(self, dataset, iterations):
        #weights = np.ones(dataset.shape[0])  # 0 should be the number of rows
        classifiers = []
        classifier_weights = []
        data_weights = [] # these 

        data_weights.append(self.init_weight)  # add the initial weights 
        
        for idx in range(iterations):

            classifier = dt.build_tree(self.exampleset, ns.EntropySelector(self.exampleset), 1, 0, weights=data_weights[-1])
            #classifier = nb.NaiveBayes(dataset, 1, 3, 0, data_weights[-1])
            classifiers.append(classifier)

            pred = classifier.predict(dataset)[:,0]
            truth = self.dataset.iloc[:, -1].values # get the truth values
            error = self.squared_error(data_weights[-1], pred, truth) 

            if error == 0 or error >= 0.5:
                return [[classifier], [1.0]] # perfect classifier, or complete crap

            #correct = np.equal(pred[:,1], truth) # rounded (0, 1) predictions

            classifier_weights.append((1/2) * np.log((1-error)/error))

            truth_scale = (truth * 2) - 1
            pred_scale = (pred * 2) - 1

            #update the weights
            next_weight = data_weights[-1] * np.exp(classifier_weights[-1] * np.multiply(truth_scale, pred_scale))
            next_weight /= np.sum(next_weight)
            data_weights.append(next_weight)

            #print(len(classifiers), len(classifier_weights), len(data_weights[:-1]))

        weights = data_weights[:-1]
        return classifiers, classifier_weights #remove the last weight
    
    def nbayes_predict(self, X):
        T = len(self.classifiers)
        pred = np.zeros([T, X.shape[0], 2])
        denom = sum(self.classifier_weights)

        for idx, classifier in enumerate(self.classifiers):
            pred[idx] = classifier.predict(X) * self.classifier_weights[idx] / denom

        return np.sum(pred, axis=0)

    def adaboost_nbayes(self, dataset, iterations):
        #weights = np.ones(dataset.shape[0])  # 0 should be the number of rows
        classifiers = []
        classifier_weights = []
        data_weights = [] # these 

        data_weights.append(self.init_weight)  # add the initial weights 
        
        for idx in range(iterations):

            classifier = nb.NaiveBayes(dataset, 1, 3, 0, data_weights[-1])
            classifiers.append(classifier)

            pred = classifier.predict(dataset)[:,0]
            truth = dataset.iloc[:, -1].values # get the truth values
            error = self.squared_error(data_weights[-1], pred, truth) 

            if error == 0 or error >= 0.5:
                return [[classifier], [1.0]] # perfect classifier, or complete crap

            #correct = np.equal(pred[:,1], truth) # rounded (0, 1) predictions

            classifier_weights.append((1/2) * np.log((1-error)/error))

            truth_scale = (truth * 2) - 1
            pred_scale = (pred * 2) - 1

            #update the weights
            next_weight = data_weights[-1] * np.exp(classifier_weights[-1] * np.multiply(truth_scale, pred_scale))
            next_weight /= np.sum(next_weight)
            data_weights.append(next_weight)

            #print(len(classifiers), len(classifier_weights), len(data_weights[:-1]))

        weights = data_weights[:-1]
        return classifiers, classifier_weights #remove the last weight

    def logreg_predict(self, X):
        T = len(self.classifiers)
        pred = np.zeros([T, X.shape[0], 2])
        denom = sum(self.classifier_weights)

        for idx, classifier in enumerate(self.classifiers):
            pred[idx] = classifier.predict(X) * self.classifier_weights[idx] / denom

        return np.sum(pred, axis=0)

    def adaboost_logreg(self, dataset, iterations):
        #weights = np.ones(dataset.shape[0])  # 0 should be the number of rows
        classifiers = []
        classifier_weights = []
        data_weights = [] # these 

        data_weights.append(self.init_weight)  # add the initial weights 
        
        for idx in range(iterations):

            #classifier = nb.NaiveBayes(dataset, 1, 3, 0, data_weights[-1])
            classifier = lr.LogisticRegression(dataset, 1, data_weights[-1], epochs=120, lr=0.1)
            classifiers.append(classifier)

            pred = classifier.predict(dataset)[:,0]
            truth = self.dataset.iloc[:, -1].values # get the truth values
            error = self.squared_error(data_weights[-1], pred, truth) 

            if error == 0 or error >= 0.5:
                return [[classifier], [1.0]] # perfect classifier, or complete crap

            #correct = np.equal(pred[:,1], truth) # rounded (0, 1) predictions

            classifier_weights.append((1/2) * np.log((1-error)/error))

            truth_scale = (truth * 2) - 1
            pred_scale = (pred * 2) - 1

            #update the weights
            next_weight = data_weights[-1] * np.exp(classifier_weights[-1] * np.multiply(truth_scale, pred_scale))
            next_weight /= np.sum(next_weight)
            data_weights.append(next_weight)

            #print(len(classifiers), len(classifier_weights), len(data_weights[:-1]))

        #weights = data_weights[:-1]
        return classifiers, classifier_weights #remove the last weight

    def squared_error(self, weight, pred, truth):
        #error = 0
        error = np.sum(np.multiply(weight, np.power(np.subtract(pred, truth), 2)), axis=0) # need to figure out axis
        #for idx, entry in enumerate(pred):
        #    error += (pred[idx] - truth[idx])^2
        return error
        
if __name__ == '__main__':
    path = '../voting'
    data = utils._convert_exampleset_to_dataframe(mldata.parse_c45(path.split('/')[-1], path))
    booster = boosting(path, data, 'logreg', 2)
    out = booster.predict(data)