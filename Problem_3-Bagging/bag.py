import utils
import numpy as np
import mldata
import argparse
import os
import utils
import random
import nbayes
import logreg
import decisiontree
import copy
import nodeSelector as ns
PRNG = 12345

# import nbayes


"""
NOTE:
outputs from algorithms:
nbayes: currently array with binary predictions in second column.  output from nbayes.predict
logreg: currently array with binary predictions in second column.  output from logreg.predict
drree: currently array with binary predictions (no columns.  1D array).  currently not output.  within dtree.accuracy, guess set has predictions.  Can make helper to return this. 
"""

def crossValBag(exampleSet, validationType, algo, iterations, k=5):
    if validationType == 0:
        # 5-Fold Stratified CROSS VALIDATION
        folds = stratified_split_data(exampleSet.schema, exampleSet, k)
        print("-------",k,"- Fold Stratified Cross Validation --------")

        total_acc = []
        total_prec = []
        total_recal = []
        total_original_results = []
        total_predictions = []
        for i in range(k):
            #Create the buildSet
            buildSet = mldata.ExampleSet(schema)
            for j in range(k):
                if i!=j:
                   for example in(folds[j]):
                        buildSet.append(example) 
            print("Fold Iteration:", i)
            test = folds[i]
            class_idx = len(test[0])-1
            classifier = Bag(buildSet, validationType, algo, iterations)
            if algo != 'dtree':
                test = utils._convert_exampleset_to_dataframe(test) 
            predictions = classifier.predict(test)
            
            print("Calculating output of this fold.")
            original_results = []
            for l in range(len(test)):
                original_results.append(test.iloc[l,class_idx])    
            TruePos = 0
            TrueNeg = 0
            FalsePos = 0
            FalseNeg = 0
            for m in range(len(predictions)):
                if predictions[m][1] == 1 and original_results[m] == 1:
                    TruePos += 1
                elif predictions[m][1] == 0 and original_results[m] == 0:
                    TrueNeg += 1
                elif predictions[m][1] == 1 and original_results[m] == 0:
                    FalsePos += 1
                elif predictions[m][1] == 0 and original_results[m] == 1:
                    FalseNeg += 1
                else:
                    print("YOU MESSED UP:", i)
            assert len(predictions) == (TrueNeg + TruePos + FalseNeg + FalsePos), "...OH NO, Sum of results doesn't equal num of results..."

            total_acc.append((TrueNeg + TruePos)/(TrueNeg + TruePos + FalseNeg + FalsePos))
            print("Error for fold: " + str(1 - (TrueNeg + TruePos)/(TrueNeg + TruePos + FalseNeg + FalsePos)))
            if TruePos + FalsePos > 0:
                total_prec.append((TruePos)/(TruePos + FalsePos))
            elif TruePos + FalsePos + FalseNeg == 0:
                total_prec.append(1)
            else:
                total_prec.append(0)
            if TruePos + FalseNeg > 0:
                total_recal.append((TruePos)/(TruePos + FalseNeg))
            elif TruePos + FalsePos + FalseNeg == 0:
                total_recal.append(1)
            else:
                total_recal.append(0)
            if i == 0:
                total_predictions = predictions
                total_original_results = original_results
            else:
                total_predictions = np.concatenate((total_predictions, predictions), axis=0)
                total_original_results = np.concatenate((total_original_results,original_results), axis=0)

        #after folds are done
        TPR = []
        FPR = []
        increment = 0.1
        threshold = 1.0
        while threshold >= 0:
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for i in range(0, len(total_predictions)):
                if total_predictions[i][0] >= threshold and total_original_results[i] == 1:
                    TP += 1
                elif total_predictions[i][0] >= threshold and total_original_results[i] == 0:
                    FP += 1
                elif total_predictions[i][0] < threshold and total_original_results[i] == 1:
                    FN += 1
                elif total_predictions[i][0] < threshold and total_original_results[i] == 0:
                    TN += 1
                else:
                    print("YOU MESSED UP:", i, total_predictions[i], total_original_results[i])
            assert len(total_predictions) == (TN + TP + FN + FP), "...OH NO, pred doens't equal original..."   
            
            TPR.append(TP/(TP + FN))
            FPR.append(FP/(FP + TN))
            threshold -= increment
        print("TPR: ",TPR)
        print("FPR: ",FPR)
        
        AUR = 0.0

        for trap in range(0, len(TPR)-1):
           xDis = (FPR[trap+1]-FPR[trap])
           yDis = (TPR[trap] + TPR[trap+1]) / 2
           AUR += xDis * yDis

        if AUR < 0.5:
            print("1 - AUR used")
            AUR = 1.0 - AUR

        avg_acc = np.average(total_acc)
        avg_pre = np.average(total_prec)
        avg_rec = np.average(total_recal)

        std_acc = np.std(total_acc)
        std_pre = np.std(total_prec)
        std_rec = np.std(total_recal)

        print("===== Folds Complete =====")
        print("Average Accuracy   :", round(avg_acc,3), round(std_acc,3))
        print("Average Precision  :", round(avg_pre,3), round(std_pre,3))
        print("Average Recall     :", round(avg_rec,3), round(std_rec,3))
        print("Area Under ROC     :", round(AUR,3))

    elif validationType == 1:
        print("------- NO Cross Validation: Running on Full Example Set --------")
        #NO CROSS VALIDATION
        total_acc = 0.0
        total_prec = 0.0
        total_recal = 0.0
        test = utils._convert_exampleset_to_dataframe(exampleSet) 
        class_idx = utils._get_class_idx(test)
        classifier = NaiveBayes(exampleSet, validationType, bins, Mestimate)
        predictions = classifier.predict(test)
            
        print("Calculating output")
        original_results = []
        for l in range(len(test)):
            original_results.append(test.iloc[l,class_idx])    
        TruePos = 0
        TrueNeg = 0
        FalsePos = 0
        FalseNeg = 0
        for m in range(len(predictions)):
            if predictions[m][1] == 1 and original_results[m] == 1:
                TruePos += 1
            elif predictions[m][1] == 0 and original_results[m] == 0:
                TrueNeg += 1
            elif predictions[m][1] == 1 and original_results[m] == 0:
                FalsePos += 1
            elif predictions[m][1] == 0 and original_results[m] == 1:
                FalseNeg += 1
            else:
                print("YOU MESSED UP:", i)
        assert len(predictions) == (TrueNeg + TruePos + FalseNeg + FalsePos), "...OH NO, Sum of results doesn't equal num of results..."

        total_acc = (TrueNeg + TruePos)/(TrueNeg + TruePos + FalseNeg + FalsePos)
        total_prec = (TruePos)/(TruePos + FalsePos)
        total_recal = (TruePos)/(TruePos + FalseNeg)

        #after folds are done
        TPR = []
        FPR = []
        increment = 0.1
        threshold = 1.0
        while threshold >= 0:
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for i in range(0, len(predictions)):
                if predictions[i][0] >= threshold and original_results[i] == 1:
                    TP += 1
                elif predictions[i][0] >= threshold and original_results[i] == 0:
                    FP += 1
                elif predictions[i][0] < threshold and original_results[i] == 1:
                    FN += 1
                elif predictions[i][0] < threshold and original_results[i] == 0:
                    TN += 1
                else:
                    print("YOU MESSED UP:", i, predictions[i], original_results[i])
            assert len(predictions) == (TN + TP + FN + FP), "...OH NO, pred doens't equal original..."   
            
            TPR.append(TP/(TP + FN))
            FPR.append(FP/(FP + TN))
            threshold -= increment

        print("TPR: ",TPR)
        print("FPR: ",FPR)
        
        AUR = 0.0

        for trap in range(0, len(TPR)-1):
           xDis = (FPR[trap+1]-FPR[trap])
           yDis = (TPR[trap] + TPR[trap+1]) / 2
           AUR += xDis * yDis

        if AUR < 0.5:
            print("1 - AUR used")
            AUR = 1.0 - AUR

        print("===== Run Complete =====")
        print("Average Accuracy   :", round(total_acc,3))
        print("Average Precision  :", round(total_prec,3))
        print("Average Recall     :", round(total_recal,3))
        print("Area Under ROC     :", round(AUR,3))

    else:
        print("Incorrect validation type argument given.")

def stratified_split_data(schema, exampleSet, numFolds):
    posData = mldata.ExampleSet(schema)
    negData = mldata.ExampleSet(schema)
    np.random.seed(PRNG)
    folds = [mldata.ExampleSet(schema) for i in range(numFolds)]

    #split input by label
    for example in exampleSet:
        if example[len(example)-1] == 1:
            posData.append(example)
        else:
            negData.append(example)

    while len(posData) > 0:
        for i in range(numFolds):
            if len(posData) == 0:
                break
            x  = np.random.randint(0, len(posData)) #get random index on the input
            folds[i].append(posData[x]) #add the element to the fold
            del posData[x] #remove from the input set

    while len(negData) > 0:
        for i in range(numFolds):
            if len(negData) == 0:
                break
            x  = np.random.randint(0, len(negData)) #get random index on the input
            folds[i].append(negData[x]) #add the element to the fold
            del negData[x] #remove from the input set

    return folds



def bag(datapath, validationType, algo, iterations):
    # TODO: Do cross-val
    path = datapath
    if(os.path.isdir(path)):
        file_base = next(el for el in reversed(path.split('/')) if el)
        exampleSet = mldata.parse_c45(file_base, path)
        schema = exampleSet.schema
    bag = Bag(exampleSet, 1, algo, 10)
    predictions = bag.predict(bag.data)
    good=0
    if algo == 'dtree':
        for i in range(len(predictions)):
            label = np.asarray(bag.data.to_float())[:,-1]
            if predictions[i,1] == label[i]:
                good += 1
    else:
        for i in range(len(predictions)):
            if predictions[i,1] == bag.data.iloc[i, len(bag.data.iloc[0,:])-1]:
                good += 1
    print(good/len(predictions))

class Bag:

    def __init__(self, data, validationType, algo, iterations):
        self.validationType = validationType
        self.algo = algo
        self.iterations = iterations
        # Pre-process data into dataframe for easier bootstrap sampling
        if algo == 'dtree':
            self.data = data
        else:
            self.data = utils._convert_exampleset_to_dataframe(data)
        # Train the data
        self.predictors = self._train_models()
        
    def _train_models(self): # All parameters saved in class object
        predictors = []
        for i in range(self.iterations):
            newDataset = self._sample_datasets()
            # Set training weights to all 1 (only used in boosting)
            training_weights = np.ones(len(newDataset))
            if self.algo == "nbayes":
                # TODO: Update files to take in DataFrame instead of ExampleSet.  
                bayesPredictor = nbayes.NaiveBayes(newDataset, 1, 3, 0, training_weights)
                predictors.append(bayesPredictor)
            elif self.algo == "logreg":
                # TODO: Update files to take in DataFrame instead of ExampleSet.  
                logregPredictor = logreg.LogisticRegression(newDataset, 1, training_weights)
                predictors.append(logregPredictor)
            elif self.algo == "dtree":
                # TODO: Update files to take in DataFrame instead of ExampleSet.  
                dtreePredictor = decisiontree.build_tree(newDataset, ns.EntropySelector(newDataset), 1, 0, weights=training_weights)
                predictors.append(dtreePredictor)
        # Return trained models
        return predictors

    def predict(self, test_dataset):

        utils.flip_labels(test_dataset, 0.25, self.algo)
        predictors = self.predictors
        predictions = []
        # Make predictions
        for model in predictors:
            prediction_probabilities = model.ensemblePrediction(test_dataset)
            predictions.append(prediction_probabilities)
        # # Do majority vote
        # nClasses = -1
        # for prediction in predictions:
        #     if prediction.max() > nClasses:
        #         nClasses = prediction.max()
        # NOTE: 0th idx = probability (used for AUC), 1th idx = binary majority vote
        majority_votes = np.empty([len(predictions[0]), 2])
        for vote in range(len(predictions[0])):
            n_neg = 0
            n_pos = 0
            for pred_idx in range(len(predictions)):
                prediction = predictions[pred_idx]
                # TODO: May need to change output format of prediction array to make consistent across models
                if prediction[vote] == 0:
                    n_neg += 1
                else:
                    n_pos += 1
            majority_votes[vote, 0] = n_pos / (n_pos + n_neg)
            majority_votes[vote, 1] = round(majority_votes[vote, 0])
        return majority_votes

    def _sample_datasets(self):
        """
        This will create a new dataset of the same size as the original, except now
        filled with randomly sampled (with repitition) data form the original set
        """

        # create copy of dataset
        dataset = self.data
        if self.algo == 'dtree':
            # Uses examplesets
            return self._sample_dtree(dataset)
        else:
            # Uses DataFrames
            return self._sample_other(dataset)
        
    def _sample_other(self, dataset):
        newDataset = dataset.copy(deep=True)
        for row in range(len(dataset)):
            # For every row in new dataset, sample random row, with repitition, from original dataset
            samplingIdx = random.randint(0,len(dataset)-1)
            newDataset.iloc[row,:] = dataset.iloc[samplingIdx,:]
        return newDataset

    def _sample_dtree(self, dataset):
        newDataset = mldata.ExampleSet()
        for i in range(len(dataset)):
            samplingIdx = random.randint(0,len(dataset)-1)
            newDataset.append(dataset[samplingIdx])
        return newDataset
        
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive Bayes Implementation')
    parser.add_argument('options', nargs=4, help="The options as specified by the prompt.")
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

    algorithm = args.options[2]
    
    iterations = int(args.options[3])
    if(iterations > 0):
        print("iterations set to", iterations)
    else:
        assert 'Unable to determine iterations.'

    crossValBag(exampleSet=exampleSet, validationType=xval, algo=algorithm, iterations=iterations)