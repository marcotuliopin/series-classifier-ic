import pandas as pd
import numpy as np
import itertools
from pyts.approximation import SymbolicAggregateApproximation
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import jensenshannon


def run_sax(data, n_bins): 
    """
    Calculate SAX transformation

    -----------
    Parameters:
    data (pd.Series): Series containing the data to use in the transformation.
    n_bins (int): The number of letters to use in the Symbolic Aggregate Approximation (SAX) representation of the data. 

    -----------
    Returns:
    sax_values (np.array of string): SAX sequence.
    """

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    sax_model = SymbolicAggregateApproximation(n_bins=n_bins,strategy='normal')
    sax_values = sax_model.fit_transform(scaled_data.reshape(1, -1))
    return sax_values[0]


def create_new_representation(data, window_size, dict):
    """
    Group symbols in the pd.Series and create a new representation of lower dimensionality of the data

    -----------
    Parameters:
    data (pd.Series): Series containing the data to use in the transformation.
    window_size (int): The number of symbols to be groupped at a time.
    dict (dictionary): The encoding of each symbol to a numeric value.

    -----------
    Returns:
    new_representation (pd.Series): new data representation.
    """

    new_representation = []
    for i in range(0, len(data) - window_size + 1):
        interval = slice(i, i + window_size)
        data_window = tuple(data[interval])
        new_representation.append(dict[data_window])
    return pd.Series(new_representation)


def calculate_js_distance(data, labels):
    """
    Calculate the mean Jensenshannon distance between pairs of points for each pair of classes.

    -----------
    Parameters:
    data (pd.DataFrame): Series containing the data to use in the transformation.
    labels (pd.Series): labels of the dataset.

    -----------
    Returns:
    js_mean (pd.DataFrame): Jensen Shannon distance between samples of distinct or same class.
    """
    symbols = np.unique(data)
    classes = np.unique(labels)
    frequencies = data.apply(lambda row : calculate_frequency(row, symbols))

    js = []
    # js = []
    # do for each pair of classes
    # for c1 in classes:
    #     for c2 in classes:
    #         js_distance = []
    #         # get data from each class
    #         class_data_1 = data[labels == c1]
    #         class_data_2 = data[labels == c2]
    #         # calculate jensenshannon distance for each pair of points
    #         for i in range(len(class_data_1)):
    #             frequency_1 = calculate_frequency(class_data_1[i], symbols)
    #             for j in range(len(class_data_2)):
    #                 frequency_2 = calculate_frequency(class_data_2[j], symbols)
    #                 js_distance.append(jensenshannon(frequency_1, frequency_2))
    #         js.append([c1, c2, js_distance])
    #         print(js[-1])
    # js = pd.DataFrame(data=js, columns=['Class 1', 'Class 2', 'JS Distance'])
    return js


def calculate_frequency(data, symbols):
    """
    Calculate the frequency of each symbol in a dataset.
    """
    unique, counts = np.unique(data, return_counts=True)
    counts = counts / counts.sum()
    frequency = []
    for symbol in symbols:
        if symbol in unique:
            frequency.append(counts[np.where(unique == symbol)[0][0]])
        else:
            frequency.append(0)
    return pd.Series(frequency)


def compute_symbols_dictionary(sax_unique_values, window_size):
    """
    Compute the encoding of each symbol to a numeric value.
    
    -----------
    Parameters:
    sax_unique_values
    window_size

    -----------
    Returns:
    dict (dictionary): The encoding of each symbol to a numeric value.
    """
    l = []
    for i in range(window_size + 1):
        for combination in itertools.product(sax_unique_values, repeat=i):
            l.append(combination)
    l.sort()
    l.remove(())
    dict = {}
    for i in range(len(l)):
        dict[l[i]] = i
    return dict


def compute_transition_matrix(data, n_symbols):
    """
    Compute transition matrix

    -----------
    Parameters:
    data (pd.Series): Series containing the data to use in the transition matrix.
    n_symbols (int): The number of unique symbols in the data. 

    -----------
    Returns:
    transition_matrix (pd.Dataframe of float): Markov transition matrix.
    """
    transition_matrix = pd.DataFrame(np.zeros((n_symbols, n_symbols)), columns=np.unique(data), index=np.unique(data))
    for i in range(len(data) - 1):
        transition_matrix[data[i]][data[i + 1]] += 1
    return transition_matrix