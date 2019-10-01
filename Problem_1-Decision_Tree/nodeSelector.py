import math
import copy
import numpy as np
import pandas as pd
import re

class EntropySelector:

    def __init__(self, example_set):
        """
        Loads in the data on instantiaton and converts to a pandas.DataFrame object for runtime increase
        """

        header = []
        print("Converting parsed data to DataFrame for runtime increase")
        i = 0
        for schema_element in example_set[0].schema:
            # Add value to end of header so we can use as unique index key
            header.append(schema_element.type + str(i))
            i += 1
        example_set_vals = []
        for element in example_set.to_float():
            example_set_vals.append(list(element))
        dataframe_example_set = pd.DataFrame(example_set_vals, columns=header)
        self.example_set = dataframe_example_set


    def _get_class_idx(self, example_set):
        # Get index of class column
        keys = example_set.keys()
        # Loop backward since we are assuming class is usually last column
        for i in range(len(keys)-1, -1, -1):
            if self._is_class(keys[i]):
                class_idx = i
                break
        return class_idx

    def _get_HY(self, example_set):
        """
        Returns the entropy of the class variable (H(Y))

        :param example_set: the set of data left to partition
        :type example_set: pd.DataFrame
        :return: H(Y)
        :rtype: float
        """

        class_idx = self._get_class_idx(example_set)
        sum = 0
        for i in range(len(example_set)):
            sum += example_set.iloc[i,class_idx]
        p_pos = sum / len(example_set)
        # Calculate entropy
        if p_pos != 0 and p_pos != 1:
            return -p_pos * math.log(p_pos,2) - (1-p_pos) * math.log(1-p_pos, 2)
        else:
            return 0

    def _getHY_X_nominal(self, example_set, attr_idx, class_idx):
        """
        Returns the entropy of the class variable given attribute X, where X is nominal (H(Y|X))

        :param example_set: the set of data left to partition in float format
        :type example_set: pd.DataFrame
        :param attr_idx: Index of the attribute corresponding to X
        :type attr_idx: int
        :param class_idx: Index of the class we are predicting
        :type class_idx: int
        :return: H(Y|X) such that X is nominal
        :rtype: float
        """

        # Find number of categories
        max_cat_val = 0
        for i in range(len(example_set)):
            # Don't include Faulty Data
            if np.isnan(example_set.iloc[i,attr_idx]):
                continue
            max_cat_val = int(example_set.iloc[i,attr_idx])+1 if int(example_set.iloc[i,attr_idx])+1 > max_cat_val else max_cat_val
        # If no categories, column already selected from
        if max_cat_val == 0:
            return 1
        # Store number of each category in sums array
        sums = np.zeros(max_cat_val)
        n_pos_cat = np.zeros(max_cat_val)
        for i in range(len(example_set)):
            # Don't include Faulty Data
            if np.isnan(example_set.iloc[i,attr_idx]):
                continue
            category = int(example_set.iloc[i,attr_idx])
            sums[category] += 1
            n_pos_cat[category] += example_set.iloc[i,class_idx]
        p_pos_cat = np.divide(n_pos_cat, sums)
        # Determine entropy by summing over each category
        hY_X = 0
        for i in range(max_cat_val):
            # If sums at i is 0, the category has been removed from the dataset and is not included in calculation
            if sums[i] == 0:
                continue
            pXi = sums[i] / sum(sums)
            p_pos_Xi = n_pos_cat[i] / sums[i]
            if p_pos_Xi != 0 and p_pos_Xi != 1:
                hY_Xi = pXi * (- p_pos_Xi * math.log(p_pos_Xi, 2) - (1-p_pos_Xi) * math.log(1-p_pos_Xi, 2))
            else:
                hY_Xi = 0
            hY_X += hY_Xi
        return hY_X

    def _getHY_X_continuous(self, example_set, attr_idx, class_idx):
        """
        Returns the entropy of the class variable given attribute X, where X is continuous (H(Y|X))

        :param example_set: the set of data left to partition in float format
        :type example_set: pd.DataFrame
        :param attr_idx: Index of the attribute corresponding to X
        :type attr_idx: int
        :param class_idx: Index of the class we are predicting
        :type class_idx: int
        :return: H(Y|X) such that X is continuous
        :rtype: float
        """

        # Sort data and re-set indeces
        sorted_set = example_set.sort_values(by=example_set.keys()[attr_idx])
        sorted_set.reset_index(inplace=True, drop=True)
        # Get splits
        split_idxs = self._get_splits(sorted_set, attr_idx, class_idx)
        if not split_idxs:
            return 1, None
        # Determine highest entropy split
        best_split_idx, best_entropy = self._get_best_split(sorted_set, split_idxs, class_idx)
        # Split <= this amount
        best_split_val = (sorted_set.iloc[best_split_idx, attr_idx] + sorted_set.iloc[best_split_idx-1, attr_idx]) / 2
        return best_entropy, best_split_val

    def _get_best_split(self, sorted_set, split_idxs, class_idx):
        """
        Takes in the sorted set and all split indeces and determines the best split based on the 
        entropy gained at each split index
        """
        best_entropy = 100 # Set higher than a possible entropy
        best_split_idx = -1 # Set to impossible index
        sorted_row = sorted_set.iloc[:,class_idx]
        for split_idx in split_idxs:
            entropy_left_half = self._get_subsplit_entropy(sorted_row, 0, split_idx)
            entropy_right_half = self._get_subsplit_entropy(sorted_row, split_idx, len(sorted_set))
            if entropy_left_half + entropy_right_half < best_entropy:
                best_entropy = entropy_left_half + entropy_right_half
                best_split_idx = split_idx
        return best_split_idx, best_entropy
                
    def _get_subsplit_entropy(self, sorted_row, start_idx, end_idx):
        """
        Calculates the entropy of a subsection of a row from the sorted dataset
        """
        n = end_idx - start_idx
        n_pos = np.nansum(sorted_row[start_idx:end_idx])
        p_pos = n_pos / n
        pX = n / len(sorted_row)
        if p_pos != 0 and p_pos != 1:
            return pX * (- p_pos * math.log(p_pos, 2) - (1-p_pos) * math.log(1-p_pos, 2))
        else:
            return 0

    def _get_splits(self, sorted_set, attr_idx, class_idx):
        """
        This is the fourth version of get_splits, and works by first splitting based on heterogeneity
        in attribute value (instead of class value as in previous versions).  Then, each neighboring
        group is checked to see if the class labels change between them.  This has shown an enormous
        runtime improvement versus the previous algorithm which would first split on class, then have multiple
        edge cases to take care of splits within the same attribute value group.
        """

        split_idxs = [0]
        set_values = sorted_set.iloc[:, attr_idx]
        for i in range(1, len(set_values)):
            # Append all potential splits, regardless of whether class values change
            if set_values[i] != set_values[i-1]:
                split_idxs.append(i)
        # Now we want to only keep indeces whose class label changes along change in attr value
        set_classes = sorted_set.iloc[:, class_idx]
        good_split_idxs = []
        for i in range(1, len(split_idxs)-1):
            # Separate out two adjacent groups of attributes (each group having the same attr values)
            class1_vals = set_classes[split_idxs[i-1]:split_idxs[i]]
            class2_vals = set_classes[split_idxs[i]:split_idxs[i+1]]
            # If class label doesn't change between groups, the max/min combo will be same for each group
            if class1_vals.max() != class2_vals.min() or class1_vals.min() != class2_vals.max():
                good_split_idxs.append(split_idxs[i])
        return good_split_idxs

    def _get_HY_X(self, attr_idx, attr_type, example_set):
        """
        Determines H(Y|X) depending on if the attribute is nominal/binary or continuous
        """

        if self._is_continuous(attr_type):
            return self._getHY_X_continuous(example_set, attr_idx, self._get_class_idx(example_set))
        # Nominal case works for binary since binary is a subset of nominal
        elif self._is_nominal(attr_type) or self._is_binary(attr_type):
            return self._getHY_X_nominal(example_set, attr_idx, self._get_class_idx(example_set)), None
        else:
            raise TypeError("Bad type.  Must be CONTINUOUS, BINARY, or NOMINAL")

    def _get_HX(self, attr_idx, attr_type, example_set, split_value):
        """
        Calculates H(X) by calculating p(x) for all x in X, and summing the relative entropies(x).  Note that
        if H(X) is 0, .001 will be returned to prevent a divide by 0 error in the gain ratio calculation.  This
        should not create problems since we care about the magnitude of the information gains relative to each other.
        """
        if self._is_nominal(attr_type) or self._is_binary(attr_type):
            num_classes = 0
            for i in range(len(example_set)):
                # Don't include Faulty Data
                if np.isnan(example_set.iloc[i, attr_idx]):
                    continue
                # Find how many classes of data there are
                num_classes = int(example_set.iloc[i, attr_idx])+1 if int(example_set.iloc[i,attr_idx])+1 > num_classes else num_classes
            # Get number of examples in each category
            nX = np.zeros(num_classes)
            for i in range(len(example_set)):
                if not np.isnan(example_set.iloc[i, attr_idx]):
                    nX[int(example_set.iloc[i, attr_idx])] += 1
            # Calculate entropy
            tot_hX = 0
            for i in range(len(nX)):
                if nX[i] != 0:
                    tot_hX -= (nX[i] / sum(nX)) * math.log(nX[i] / sum(nX), 2)
            if tot_hX == 0:
                # Use fraction since 0 will raise error.
                return .001
            else:
                return tot_hX
        elif self._is_continuous(attr_type):
            # If no split, all same class
            if split_value is None:
                return .001
            # We know there are 2 classes
            classes = np.zeros(2)
            for i in range(len(example_set)):
                # Determine number in each of the two classes
                if not np.isnan(example_set.iloc[i, attr_idx]) and example_set.iloc[i, attr_idx] <= split_value:
                    classes[0] += 1
                elif not np.isnan(example_set.iloc[i, attr_idx]) and example_set.iloc[i, attr_idx] > split_value:
                    classes[1] += 1
            tot_hX = 0
            for i in range(2):
                # Calculate total entropy
                if classes[i] != 0:
                    tot_hX -= (classes[i] / sum(classes)) * math.log(classes[i] / sum(classes), 2)
            if tot_hX == 0:
                # Use fraction since 0 will raise error.
                return .001
            else:
                return tot_hX

    def get_split_attr(self, filter_conditions, split_criterion, return_majority_class=False):
        """
        Uses current examples to find highest entropy category

        :return: Column index of the highest entropy category
        :rtype: int
        """
        print("Dataset Filtering")
        example_set = copy.deepcopy(self.example_set)
        unusable_attr_idxs = self._filter_dataset(example_set, filter_conditions)
        # If depth reached, majority class of filtered dataset is returned
        if return_majority_class:
            class_idx = self._get_class_idx(example_set)
            return -1, round(np.nansum(example_set.iloc[:,class_idx]) / len(example_set.iloc[:,class_idx]))
        # Find how many attributes to check
        attrIdxs = []
        attrTypes = []
        # If data is missing for example, return "None" as flag for that.
        if len(example_set) == 0:
            print("Missing Data")
            return -1, None
        for i in range(len(example_set.iloc[0,:])):
            if self._is_binary(example_set.keys()[i]) or self._is_nominal(example_set.keys()[i]) or self._is_continuous(example_set.keys()[i]):
                if i not in unusable_attr_idxs:
                    attrIdxs.append(i)
                    attrTypes.append(example_set.keys()[i])
        information_gains = []
        split_values = []
        # Get entropies
        print("Calculating entropies (" +  str(len(attrTypes)-1) + ")")
        for i in range(len(attrTypes)):
            entropy, split_value = self._get_HY_X(attrIdxs[i], attrTypes[i], example_set)
            info_gain = self._get_HY(example_set) - entropy
            # Use gain ratio if split_criterion == 1
            if split_criterion == 1:
                info_gain /= self._get_HX(attrIdxs[i], attrTypes[i], example_set, split_value)
            information_gains.append(info_gain)
            split_values.append(split_value)
        # If no gain from any attribute, specify to stop making tree and return class value (homogeneous)
        if np.max(information_gains) <= 0:
            return -1, example_set.iloc[0,self._get_class_idx(example_set)]
        bestAttrIdx = attrIdxs[np.argmax(information_gains)]
        bestSplitValue = split_values[np.argmax(information_gains)]
        print("Information gain of best attribute: " + str(max(information_gains)))
        return bestAttrIdx, bestSplitValue

    def _filter_dataset(self, example_set, filter_conditions):
        """
        This will remove all rows in the dataset that have attributes which are conditioned on.
        """

        unusable_attr_idxs = []
        # Retrieve the attribute indeces and values for filtering
        filter_attr_idxs = list(map(int, filter_conditions.keys()))
        filter_values = list(filter_conditions.values())
        for filter_attr_idx in filter_attr_idxs:
            # Cannot re-use selected attributes unless continuous
            if not self._is_continuous(example_set.keys()[filter_attr_idx]):
                unusable_attr_idxs.append(filter_attr_idx)
        # Loop backward, popping off rows
        drop_indeces = []
        for i in range(len(example_set)-1, -1, -1):
            # Get rid of all rows that don't fit conditions
            for j in range(len(filter_attr_idxs)):
                attr_type = example_set.keys()[filter_attr_idxs[j]]
                value = example_set.iloc[i,filter_attr_idxs[j]]
                if self._is_nominal(attr_type) or self._is_binary(attr_type):
                    if value != filter_values[j]:
                        drop_indeces.append(i)
                        # Move to next example
                        break
                elif self._is_continuous(attr_type):
                    # Check >= and <=
                    conditionals = filter_values[j]
                    # Left bound exists
                    if conditionals[0] is not None:
                        if value <= conditionals[0]:
                            drop_indeces.append(i)
                            break
                    if conditionals[1] is not None:
                        if value > conditionals[1]:
                            drop_indeces.append(i)
                            break
        example_set.drop(drop_indeces, inplace=True)
        # Reset indices for later indexing
        example_set.reset_index(inplace=True, drop=True)
        return unusable_attr_idxs

    def _is_continuous(self, type):
        pattern = re.compile("CONTINUOUS*")
        return bool(pattern.match(type))

    def _is_nominal(self, type):
        pattern = re.compile("NOMINAL*")
        return bool(pattern.match(type))

    def _is_binary(self, type):
        pattern = re.compile("BINARY*")
        return bool(pattern.match(type))

    def _is_class(self, type):
        pattern = re.compile("CLASS*")
        return bool(pattern.match(type))
        


        


class TypeError(Exception):
    pass



class FunctionNotSupported(Exception):
    pass


if __name__ == '__main__':
    # TESTING --- Delete for finished product
    import sys
    sys.path.append("..")
    import mldata
    x = mldata.parse_c45("spam", "../spam")
    e = EntropySelector(x)
    print(e.get_split_attr({6: 0.0}, 0))