from abc import ABC, abstractmethod
from enum import Enum, auto
import mldata
import nodeSelector as ns
import queue
import math

DEBUG = True

class DecisionTree():
    """
    The purpose of the DecisionTree object is to contain a reference to the "head" node, 
    which also contains the attribute with the highest information gain. This class also
    values for the max depth and length of the tree.
    """

    def __init__(self, headnode, maxdepth):
        """
        Simple constructor for DecisionTree.  When initializing DecisionTree, a head node
        MUST be specified. (Why would you want to initialize a DecisionTree otherwise?)

        :param headnode: The node containing the first attribute to make a decision with.
        :type headnode: AbstractNode
        :param maxdepth: The maximum allowed depth of the DecisionTree
        :type maxdepth: int
        :return: An initialized DecisionTree
        :rtype: DecisionTree
        """
        self.headnode = headnode
        self.maxdepth = maxdepth
        self.depth = 0
        self.size = 1
        self.unexplored = queue.Queue()
 
    def eval(self, vector):
        """
        Eval will output the classification of an input vector that is properly formed for
        the DecisionTree.  If the input vector is not properly formed, it will not work. 

        :param vector: A vector containing the attribute values to be evaluated.
        :type vector: mldata.Example
        :return: The label, as evaluated by the constructed DecisionTree
        :rtype: float
        """
        node = self.headnode
        while not isinstance(node, ClassNode):
            node = node._eval(vector[node.index])
        return node.attr_float

    def eval_set(self, vector_set):
        """
        Eval_set will output the estimated value for each vector in an ExampleSet

        :param vector_set: A set of vectors to be evaluate
        :type vector_set: mldate.ExampleSet
        :return: A list of evaluations for each vector.
        :rtype: List of floats
        """
        guess = []
        for vector in vector_set:
            guess.append(self.eval(vector))
        return guess

    # TODO: Make this function for mental extra credit.
    def prune(self, verification):
        """
        This function will use a dataset, verification, and use the information gains of the
        attributes in that dataset to prune the tree.  Nodes will be removed if they increase
        information gain.

        :param verification: A verification dataset that is independent from the training set.
        :type verification: mldata.ExampleSet
        """
        print('placeholder')

class AbstractNode():
    def __init__(self, parent, name, values, index, attr_float, key):
        """
        All AbstractNodes are initialized with the name of the attribute, and the attribute labels
        that are provided by the data parser.

        :param name: This is the name of the attribute, as provided by the data set.
        :type name: str
        :param values: A dictionary of classification to child
        :type values: Dict
        :return: An initialized object of type AbstractNode
        :rtype: AbstractNode
        """
        self.parent = parent
        self.name = name
        self.values = values
        self.index = index
        self.attr_float = attr_float
        self.key = key

    @abstractmethod
    def _eval(self, attr_float):
        """
        This provide the child node that corresponds with the attribute value 
        :param feature: The value of the attribute classified by this node
        :type feature: varies
        :return: Either the classification, if this is a leaf node, or a child node
        :rtype: varies
        """
        pass

class BinaryNode(AbstractNode):
    def __init__(self, parent, name, values, index, attr_float, key):
        super().__init__(parent, name, values, index, attr_float, key)
        self.values = {0:None, 1:None}

    def _eval(self, attr_float):
        return self.values[attr_float]
        
class NominalNode(AbstractNode):
    def __init__(self, parent, name, values, index, attr_float, key):
        super().__init__(parent, name, values, index, attr_float, key)
        self.values = {elem:None for elem in values}

    def _eval(self, attr_float):
        return self.values[attr_float]

class ContinuousNode(AbstractNode):
    def __init__(self, parent, name, values, index, attr_float, key):
        super().__init__(parent, name, values, index, attr_float, key)
        self.values = {(None, self.attr_float):None, (self.attr_float, None):None}

    def _eval(self, attr_float):
        if attr_float <= self.attr_float:
            return self.values[(None, self.attr_float)]
        elif attr_float > self.attr_float:
            return self.values[(self.attr_float, None)]

class ClassNode(AbstractNode):
    def __init__(self, parent, name, values, index, attr_float, key):
        super().__init__(parent, name, values, index, attr_float, key)

    def _eval(self, attr_float):
        return self.values

def _init_node(x, parent, attr_idx, attr_float, key):
    """
    The purpose of this function is the create a new node for a specific attribute,
    identified by the index split_atr.  This index is provided by Jeff's super sweet
    get_split_attr function, that abstracts all the actual math from me.  After getting
    the type, name, and values of the attribute in question, the correct type of node
    is initialized, with the name and values provided by the dataset.  Will throw an
    error for attributes that I haven't specifically defined.

    :param x: Either a single row, or the full dataset that is being used.
    :type x: mldata.ExampleSet
    :param split_atr: The index of the attribute with the highest information gain.
    :type split_atr: int
    :return: Returns an initialized node of the attribute type.
    :rtype: AbstractNode
    """

    if(DEBUG):
        print('[DEBUG] Type:',x.schema[attr_idx].type, '- Name:', x.schema[attr_idx].name)

    if x.schema[attr_idx].type is 'BINARY':
        return BinaryNode(parent, x.schema[attr_idx].name, x.schema[attr_idx].values, attr_idx, attr_float, key)
    elif x.schema[attr_idx].type is 'NOMINAL':
        return NominalNode(parent, x.schema[attr_idx].name, x.schema[attr_idx].values, attr_idx, attr_float, key)
    elif x.schema[attr_idx].type is 'CONTINUOUS':
        return ContinuousNode(parent, x.schema[attr_idx].name, x.schema[attr_idx].values, attr_idx, attr_float, key)
    elif x.schema[attr_idx].type is 'CLASS':
        return ClassNode(parent, x.schema[attr_idx].name, x.schema[attr_idx].values, attr_idx, attr_float, key)
    else:
        assert 'Data type not supported.'

def build_tree(dataset, entropy_selector, maxdepth, split_crit):
    """
    This function will iteratively build a tree, in the order of information gain,
    as provided by nodeSelector.  

    :param dataset: The dataset which the tree is built from.
    :type dataset: mldata.ExampleSet
    :param entropy_selector: An nodeSelector initialized with the correct dataset.
    :type entropy_selector: nodeSelector
    :param maxdepth: Specifies the deepest path on the decision tree.
    :type maxdepth: int
    :param split_crit: Information Gain (0), Gain Ratio (1)
    :type split_crit: int
    :return: returns an initialized decisiontree on a specific dataset
    :rtype: DecisionTree
    """
    head, depth = max_IG_node(dataset, None, None, maxdepth, entropy_selector, split_crit)
    decision_tree = DecisionTree(head, maxdepth)
    if(decision_tree.depth < depth):
        decision_tree.depth = depth
    if(head.index > -1):
        decision_tree.unexplored.put(head)
    while not decision_tree.unexplored.empty():
        to_expand = decision_tree.unexplored.get()
        decision_tree.size += len(to_expand.values)
        for key in to_expand.values:
            to_expand.values[key], depth = max_IG_node(dataset, to_expand, key, decision_tree.maxdepth, entropy_selector, split_crit)
            if(decision_tree.depth < depth):
                decision_tree.depth = depth
            if(to_expand.values[key].index > -1):
                decision_tree.unexplored.put(to_expand.values[key])
    return decision_tree

def max_IG_node(dataset, to_expand, key, maxdepth, entropy_selector, split_crit):
    """
    This function returns the node with the maximum information gain, 
    specified by nodeSelector

    :param dataset: The dataset which the tree is built from.
    :type dataset: mldata.ExampleSet
    :param to_expand: The node that is being expanded
    :type to_expand: AbstractNode
    :param key: The attribute value in question.
    :type key: varies
    :param maxdepth: Specifies the deepest path on the decision tree.
    :type maxdepth: int
    :param entropy_selector: An nodeSelector initialized with the correct dataset.
    :type entropy_selector: nodeSelector
    :param split_crit: Information Gain (0), Gain Ratio (1)
    :type split_crit: int
    :return: returns an initialized Node, of the correct type.
    :rtype: AbstractNode
    """
    decisions = _trace(dataset, to_expand, key)
    depth = len(decisions)
    truncate = maxdepth != -0 and depth >= maxdepth
    attr_idx, attr_float = entropy_selector.get_split_attr(decisions, split_crit, truncate)
    return _init_node(dataset, to_expand, attr_idx, attr_float, key), depth

def _trace(dataset, node, value):
    """
    This function returns the attribute values required to get to a specific node
    in backwards order, since it doesn't matter.

    :param dataset: The dataset which the tree is built from.
    :type dataset: mldata.ExampleSet
    :param node: The node for which the attribute values are in question
    :type node: AbstractNode
    :param value: This is that last key in question for the trace
    :type value: variable
    :return: A dictionary of attribute values required to get to node
    :rtype: dict
    """
    attrs = []
    if (node is not None):
        attrs.append([node.index, dataset.schema[node.index].to_float(value)])
    
    while(node is not None and node.parent is not None):
        attrs.append([node.parent.index, dataset.schema[node.parent.index].to_float(node.key)])
        node = node.parent
    attrs = _combine_terms(attrs)
    trace = {elem[0]:elem[1] for elem in attrs}

    if(DEBUG):
        print("[DEBUG] Trace:", trace)
    return trace

def _combine_terms(attrs):
    """
    The purpose of this function is properly format multiple nodes of continuous variables,
    which effectively define a range within that continuous variable.

    :param attrs: A list of attributes in a trace.
    :type attrs: List
    :return: A list of attributes where continuous variables of the same type are collected.
    :rtype: List
    """
    for i in range(len(attrs)):
        for j in range(len(attrs)):
            if (i != j and attrs[i][0] == attrs[j][0]):
                split1 = attrs[i][1]
                split2 = attrs[j][1]
                attrs.remove(attrs[j])
                attrs[i][1] = [next(elem for elem in split1 if elem is not None), next(elem for elem in split2 if elem is not None)]
                attrs[i][1].sort()
                return attrs
    return attrs

if __name__ == '__main__':
    # This code is for testing purposes.
    x = mldata.parse_c45('voting', '../voting')
    #e = ns.EntropySelector(x)
    #dtree = build_tree(x, e, 0)
    #print(x[0].to_float())
    #print(dtree.eval(x[0]).attr_float)
    #print(x[3])
    #print(dtree.eval(x[3]).attr_float)
    #print(_combine_terms([[1,4.5],[3,(None,1234)], [6, "AY"], [3,(54,None)], [4, 4.0]]))
    #print(e.get_split_attr({2:4.0}, 0))
    #attr_idx = -1
    #attr_idx, attr_float = e.get_split_attr({2:4.0}, 0)
    #print(attr_idx, attr_float)
    #node = _init_node(x, None, attr_idx, attr_float, attr_idx)
