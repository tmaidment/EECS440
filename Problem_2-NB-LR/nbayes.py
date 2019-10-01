
import utils
import numpy as np
import mldata
import argparse
import os
PRNG = 12345

def nbayes(schema,exampleSet, validationType, bins, Mestimate, k=5):
    if validationType == 0:
        # 5-Fold Stratified CROSS VALIDATION
        folds = stratified_split_data(schema, exampleSet, k)
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
            test = utils._convert_exampleset_to_dataframe(folds[i]) 
            class_idx = utils._get_class_idx(test)
            classifier = NaiveBayes(buildSet, validationType, bins, Mestimate)
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

class NaiveBayes:

    def __init__(self, data, validationType, bins, mEstimate):
        self.validationType = validationType
        self.bins = bins
        self.mEstimate = mEstimate
        proc_data = utils._convert_exampleset_to_dataframe(data)
        # Convert all continuous attributes to classes
        self._convert_data(proc_data, bins)
        # Get P(Xi = xi | Y = y) for all Xi, xi, and y.  Specific data structure
        # storage details are discussed in helper methods
        data_pred_pos_outcome, data_pred_neg_outcome = self._get_pos_neg_outcomes(proc_data, mEstimate)
        # Store probabilities of attributes
        self.pos_outcomes = data_pred_pos_outcome
        self.neg_outcomes = data_pred_neg_outcome
        # Store class probability
        class_idx = utils._get_class_idx(proc_data)
        class_data = proc_data.iloc[:,class_idx]
        self.p_pos_class = self._get_n_class(class_data, 1) / len(class_data)

    def _get_pos_neg_outcomes(self, proc_data, m):
        """
        Saves the probabilities of an attribute being a value, given outcome 0 or 1.  The 
        format of how the data is saved is explained below above the dict instantiations
        """

        print("Starting probability estimates")
        # This will store as key the attribute index, and have value be list of probabilities, with
        # each index being the attr value and each value being the probability of that attr value given positive class
        data_pred_pos_outcome = {}
        # This will store as key the attribute index, and have value be list of probabilities, with
        # each index being the attr value and each value being the probability of that attr value given negative class
        data_pred_neg_outcome = {}
        class_idx = utils._get_class_idx(proc_data)
        for attr_idx in range(len(proc_data.iloc[0,:])):
            p_data_given_y0 = self._p_data_given_y(proc_data, attr_idx, class_idx, 0, m)
            p_data_given_y1 = self._p_data_given_y(proc_data, attr_idx, class_idx, 1, m)
            # If returns are -1, then attribute is a class or index col., which we don't want
            if p_data_given_y0 is not None and p_data_given_y1 is not None:
                # Store list of probabilities corresponding to attr val under dict key of attr idx
                data_pred_pos_outcome[str(attr_idx)] = p_data_given_y1
                data_pred_neg_outcome[str(attr_idx)] = p_data_given_y0
        return data_pred_pos_outcome, data_pred_neg_outcome
        

    def _p_data_given_y(self, data, attr_idx, class_idx, y_value, m):
        """
        TODO: Add in m-Estimate values
        This will calculate the probability of x given y.
        P(Xi=xi|Y=y) = (# examples with Xi=xi and Y=y) / (# examples with Y=y)
        """

        # Make sure attribute isn't of type ID or CLASS
        attr_type = data.keys()[attr_idx]
        if not (utils._is_continuous(attr_type) or utils._is_nominal(attr_type) or utils._is_binary(attr_type)):
            return None
        # Get class data and attr data
        class_data = data.iloc[:,class_idx]
        attr_data = data.iloc[:,attr_idx]
        n_attr_vals = int(attr_data.max()) + 1 # Indexing starts at 0 for each different value (all categorical after numeric is converted)
        # Extract probs
        nX_and_Y = np.zeros(n_attr_vals)
        for i in range(len(attr_data)):
            if class_data[i] == y_value:
                # Make sure attribute is not NaN
                if not np.isnan(attr_data[i]):
                    nX_and_Y[int(attr_data[i])] += 1
        p = 1/n_attr_vals
        # Laplace Smoothing
        if m < 0:
            m = n_attr_vals
        p_data_given_y = (nX_and_Y + m*p) / (self._get_n_class(class_data, y_value) + m)
        return p_data_given_y


    def _get_n_class(self, class_data, class_label):
        if class_label == 0:
            return len(class_data) - class_data.sum()
        else:
            return class_data.sum()

    def _convert_data(self, data, bins):
        print("Converting continuous values to bins")
        # Type of data stored as table keys
        column_types = list(data.keys())
        for col in range(len(data.iloc[0,:])):
            # If column continuous, convert values in column to a bin index
            if utils._is_continuous(column_types[col]):
                # Get minimum and maximum element for bins
                _min = np.amin(data.iloc[:,col])
                _max = np.amax(data.iloc[:,col])
                cutoffs = np.linspace(_min, _max, bins+1)
                # Replace each continuous element with bin number
                for row in range(len(data)):
                    value = data.iloc[row,col]
                    for bin_idx in range(0, len(cutoffs)-1):
                        if value <= cutoffs[bin_idx+1]:
                            data.at[row, column_types[col]] = bin_idx
                            break
        # Returns nothing since modification is in-place

    def predict(self, X):
        """
        Return the predicted probability that the class label is +1 given X.
        NOTE: Assumes X is the whole prediction set
        param X: examples of type pandas.Dataframe
        returns PR[Y=1|X] as a 1D array with same number of values as rows in Dataframe
        """

        print("Starting predictions")
        # Convert continuous vals for prediction
        self._convert_data(X, self.bins)
        prediction_probabilities = np.zeros((len(X),2))
        # Extract which columns are valid (not class or ID)
        attr_is_valid = np.zeros(len(X.iloc[0,:]))
        keys = X.keys()
        for i in range(len(X.iloc[0,:])):
            attr_type = keys[i]
            attr_is_valid[i] = utils._is_continuous(attr_type) or utils._is_nominal(attr_type) or utils._is_binary(attr_type)
        # Compute probabilities
        for i in range(len(X)):
            feature_vec = X.iloc[i,:]
            # Left idx = P(D|y=0), right idx = Pr(D|y=1).  Initial value is p(Y=y)
            p_data_given_y = np.array([1-self.p_pos_class, self.p_pos_class])
            for j in range(len(feature_vec)):
                if attr_is_valid[j]:
                    # NOTE: Currently if NaN present, doesn't include it
                    if not np.isnan(feature_vec[j]):
                        p_data_given_y[0] *= self.neg_outcomes[str(j)][int(feature_vec[j])]
                        p_data_given_y[1] *= self.pos_outcomes[str(j)][int(feature_vec[j])]
            #print(p_data_given_y)
            if p_data_given_y[1] + p_data_given_y[0] == 0:
                # print("oh no! - p_data_given_y: ", p_data_given_y)
                prediction_probabilities[i][0] = 0.0
            else: 
                prediction_probabilities[i][0] = p_data_given_y[1] / (p_data_given_y[1] + p_data_given_y[0])
            prediction_probabilities[i][1] = np.argmax(p_data_given_y)
        return prediction_probabilities




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

    NumBins = int(args.options[2])
    if(NumBins > 0):
        print("Number of Bins set to", NumBins)
    elif(NumBins == 0):
        print("No Number of Bins value set.")
    else:
        assert 'Negative Bins is not supported.'

    MEstimate = int(args.options[3])
    if(MEstimate == 0):
        print("M Estimate M value set to MLE")
    elif(MEstimate < 0):
        print("M Estimate M value set to Laplace smoothing", MEstimate)
    elif(MEstimate > 0):
        print("M Estimate M value set to", MEstimate)
    else:
        assert 'Unable to determine MEstimate.'

    nbayes(schema, exampleSet, validationType=xval, bins=NumBins, Mestimate=MEstimate, k=5)