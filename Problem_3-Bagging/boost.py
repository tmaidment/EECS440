import utils
import numpy as np
import mldata
import argparse
import os
import boosting as b
PRNG = 12345

def boost(exampleSet, validationType, algo, iterations, k=5):
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
            classifier = b.boosting(buildSet, algo, iterations)
            if algo != 'dtree':
                test2 = utils._convert_exampleset_to_dataframe(test) 
                predictions = classifier.predict(test2)
            else:
                #test = exampleSet
                predictions = classifier.predict(test)
                #print(predictions)

            test = utils._convert_exampleset_to_dataframe(test) 
            
            print("Calculating output of this fold.")
            original_results = []
            for l in range(len(test)):
                original_results.append(test.iloc[l,class_idx])    
            TruePos = 0
            TrueNeg = 0
            FalsePos = 0
            FalseNeg = 0
            for m in range(len(predictions)):
                predictions[m][1] = round(predictions[m][1])
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
                    #print(len(predictions), predictions[m][1])
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
        classifier = b.boosting(exampleSet, algo, iterations)
        if algo != 'dtree':
            test2 = utils._convert_exampleset_to_dataframe(exampleSet) 
            predictions = classifier.predict(test2)
        else:
            #test = exampleSet
            predictions = classifier.predict(exampleSet)
            #print(predictions)
            
        print("Calculating output")
 
        original_results = []
        for l in range(len(test)):
            original_results.append(test.iloc[l,class_idx])   

        TruePos = 0
        TrueNeg = 0
        FalsePos = 0
        FalseNeg = 0
        for m in range(len(predictions)):
            predictions[m][1] = round(predictions[m][1])
            if predictions[m][1] == 1 and original_results[m] == 1:
                TruePos += 1
            elif predictions[m][1] == 0 and original_results[m] == 0:
                TrueNeg += 1
            elif predictions[m][1] == 1 and original_results[m] == 0:
                FalsePos += 1
            elif predictions[m][1] == 0 and original_results[m] == 1:
                FalseNeg += 1
            else:
                print("YOU MESSED UP:", m)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Boosting Implementation')
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

    boost(exampleSet=exampleSet, validationType=xval, algo=algorithm, iterations=iterations)