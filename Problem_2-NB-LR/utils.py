"""
This file contains all functions shared by the logreg and nbayes.
This is primarily used to keep data formatting consistent.
"""
import pandas as pd
import re

def _convert_exampleset_to_dataframe(example_set):
    """
    Converts the double linked list to a DataFrame, with the 
    types of each attribute being column headers.
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
    return dataframe_example_set

def _is_continuous(type):
    pattern = re.compile("CONTINUOUS*")
    return bool(pattern.match(type))

def _is_nominal(type):
    pattern = re.compile("NOMINAL*")
    return bool(pattern.match(type))

def _is_binary(type):
    pattern = re.compile("BINARY*")
    return bool(pattern.match(type))

def _is_class(type):
    pattern = re.compile("CLASS*")
    return bool(pattern.match(type))

def _get_class_idx(example_set):
    # Get index of class column
    keys = example_set.keys()
    # Loop backward since we are assuming class is usually last column
    for i in range(len(keys)-1, -1, -1):
        if _is_class(keys[i]):
            class_idx = i
            break
    return class_idx