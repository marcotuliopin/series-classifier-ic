import pandas as pd
import numpy as np
import itertools
from pyts.approximation import SymbolicAggregateApproximation
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import jensenshannon
from itertools import combinations


def run_sax(data, n_bins): 
    """
    Calculate SAX transformation

    Parameters:
    -----------
    data (pd.Series): Series containing the data to use in the transformation.
    n_bins (int): The number of letters to use in the Symbolic Aggregate Approximation (SAX) representation of the data. 

    Returns:
    -----------
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

    Parameters:
    -----------
    data (pd.Series): Vector containing the data to use in the transformation.
    window_size (int): The number of symbols to be groupped at a time.
    dict (dictionary): The encoding of each symbol to a numeric value.

    Returns:
    -----------
    new_representation (pd.Series): Vector containing the new data representation.
    """

    new_representation = []
    for i in range(0, len(data) - window_size + 1):
        interval = slice(i, i + window_size)
        data_window = tuple(data[interval])
        new_representation.append(dict[data_window])
    return pd.Series(new_representation)


def calculate_js_distance(data):
    """
    Calculate the Jensenshannon distance between pairs of points for each pair of classes.

    Parameters:
    -----------
    data (pd.DataFrame): Matrix containing the data to use in the transformation.

    Returns:
    -----------
    js_mean (pd.DataFrame): Jensen Shannon distance between samples' pairs.
    """
    symbols = np.unique(data)
    frequencies = data.apply(lambda row : calculate_frequency(row, symbols))

    # Calculate Jensenshannon distance for each pair
    pairwise_js = np.zeros((len(frequencies), len(frequencies)))
    pairwise_js[np.triu_indices(len(frequencies), k=1)] = [jensenshannon(x, y) for x, y in combinations(frequencies.values, r=2)]
    pairwise_js += pairwise_js.T
    pairwise_js = pairwise_js

    return pairwise_js


def get_js_by_class(pairwise_js, labels):
    """
    Given a square matrix of Jensenshannon distances for each pair of samples, returns a vector of Jensenshannon
    distances for pairs of samples from the same class and another for pairs from different classes.

    Parameters:
    -----------
    pairwise_js (pd.DataFrame): Matrix of Jensenshannon distances for each pair of samples.
    labels (pd.Series): Vector containing the label of each sample.

    Returns:
    -----------
    eq_class_js (pd.Series): Vector of Jensenshannon distances for pairs from the same class.
    diff_class_js (pd.Series): Vector of Jensenshannon distances for pairs of from different classes.
    """
    eq_class_js = np.ones(pairwise_js.shape) * -1
    diff_class_js = np.ones(pairwise_js.shape) * -1

    for i in range(pairwise_js.shape[0]):
        indexes = (labels[i + 1:] == labels[i])
        p_tmp = pairwise_js[i, i + 1:]
        # Get JS distance from instances that belong to the same class
        eq_tmp = eq_class_js[i, i + 1:]
        eq_tmp[indexes] = p_tmp[indexes]
        # Get JS distance from instances that belong to the different classes
        diff_tmp = diff_class_js[i, i + 1:]
        diff_tmp[~indexes] = p_tmp[~indexes]

    eq_class_js = eq_class_js.flatten()
    eq_class_js = pd.Series(eq_class_js[np.where(eq_class_js != -1)])

    diff_class_js = diff_class_js.flatten()
    diff_class_js = pd.Series(diff_class_js[np.where(diff_class_js != -1)])

    return eq_class_js, diff_class_js


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

    frequency = pd.Series(frequency)
    return frequency


def compute_symbols_dictionary(sax_unique_values, window_size):
    """
    Compute the encoding of each symbol to a numeric value.
    
    Parameters:
    -----------
    sax_unique_values
    window_size

    Returns:
    -----------
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

    Parameters:
    -----------
    data (pd.Series): Series containing the data to use in the transition matrix.
    n_symbols (int): The number of unique symbols in the data. 

    Returns:
    -----------
    transition_matrix (pd.Dataframe of float): Markov transition matrix.
    """
    transition_matrix = pd.DataFrame(np.zeros((n_symbols, n_symbols)), columns=np.unique(data), index=np.unique(data))
    for i in range(len(data) - 1):
        transition_matrix[data[i]][data[i + 1]] += 1
    return transition_matrix