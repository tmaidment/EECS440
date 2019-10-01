import mldata
import decisiontree as dt
import numpy as np
import nodeSelector as ns
from itertools import cycle
import argparse
import os

PRNG = 12345

def dtree(exampleSet, validationType, depth, splitCriterion, k=5):
    e = ns.EntropySelector(exampleSet)

    if validationType == 0:
        # 5-Fold Stratified CROSS VALIDATION
        folds = stratified_split_data(exampleSet, k)
        print("-------",k,"- Fold Stratified Cross Validation --------")

        total_acc = 0

        for i in range(k):
            #Create the buildSet
            buildSet = mldata.ExampleSet()
            for j in range(k):
                if i!=j:
                    buildSet.append(folds[j]) 
            #Build tree and output for each fold     
            #print(buildSet)
            tree = dt.build_tree(buildSet, e, depth, splitCriterion)  
            acc = accuracy(tree, folds[i])         
            print("Fold Iteration:", i)
            print("Accuracy     :", acc)
            print("Size         :", tree.size)
            print("Maximum Depth:", tree.depth)
            print("First Feature:", tree.headnode.name)
            total_acc += acc

        print("Average Accuracy:", total_acc/k)
        

    elif validationType == 1:
        print("------- NO Cross Validation: Running on Full Example Set --------")
        #NO CROSS VALIDATION
        tree = dt.build_tree(exampleSet, e, depth, splitCriterion)
        print("Accuracy     :", accuracy(tree, exampleSet))
        print("Size         :", tree.size)
        print("Maximum Depth:", tree.depth)
        print("First Feature:", tree.headnode.name)
    else:
        print("Incorrect validation type argument given.")


def stratified_split_data(exampleSet, numFolds):
    posData = mldata.ExampleSet()
    negData = mldata.ExampleSet()
    np.random.seed(PRNG)
    folds = [mldata.ExampleSet() for i in range(numFolds)]

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

def accuracy(tree, testSet):
    """
    Calculates the accuracy of a test set on an initialized DecisionTree

    :param tree: An initialized DecisionTree, that has been fully populated.
    :type tree: DecisionTree
    :param testSet: A independent set of data to test on the tree.
    :type testSet: mldata.ExampleSet
    :return: The accuracy of the testSet on the decision tree.
    :rtype: float
    """
    guess = np.asarray(tree.eval_set(testSet))
    label = np.asarray(testSet.to_float())[:,-1]
    true = np.count_nonzero(guess == label)
    false = np.count_nonzero(guess != label)

    return true / (true + false)

if __name__ == '__main__':

    '''
    path = '../spam'
    x = mldata.parse_c45(path.split('/')[-1], path)
    dtree(x, validationType=0, depth=5, splitCriterion=1)
    '''

    parser = argparse.ArgumentParser(description='ID3 Decision Tree Implementation')
    parser.add_argument('options', nargs=4, help="The options as specified by the prompt.")
    args = parser.parse_args()

    path = str(args.options[0])
    if(os.path.isdir(path)):
        exampleSet = mldata.parse_c45(next(el for el in reversed(path.split('/')) if el), path)
        print("Loading dataset:", next(el for el in reversed(path.split('/')) if el))
    else:
        assert 'Dataset input not found!'

    xval = int(args.options[1])
    if(xval == 0):
        print("Cross Validation enabled")
    elif(xval == 1):
        print("Cross Validation disabled")
    else:
        assert 'Unable to determine cross validation flag.'

    maxdepth = int(args.options[2])
    if(maxdepth > 0):
        print("Maxdepth set to", maxdepth)
    elif(maxdepth == 0):
        print("No maxdepth value set.")
    else:
        assert 'Negative maxdepth is not supported.'

    split_criterion = int(args.options[3])
    if(split_criterion == 0):
        print("Information Gain used as split criterion")
    elif(split_criterion == 1):
        print("Gain Ratio used as split criterion")
    else:
        assert 'Unable to determine split criterion flag.'

    dtree(exampleSet, validationType=xval, depth=maxdepth, splitCriterion=split_criterion)