import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def flatten(array):
    if isinstance(array, list):
        if len(array) > 1 and isinstance(array[0], list):
            array = np.array([val for sublist in array for val in sublist])
        return array
    if len(array.shape) == 1:
        return array
    else:
        return np.array([val for sublist in array for val in sublist])


def unique_count_data(labels):
    return np.unique(flatten(labels), return_counts=True)


def compute_class_weights(labels, method):
    unique, counts = unique_count_data(labels)
    if method == 'log':
        return np.array([abs(np.log(x / sum(counts))) for x in counts])
    elif method == 'avg':
        return np.array([1 - (x / sum(counts)) for x in counts])
    elif method == 'scikit':
        return compute_class_weight('balanced', unique, flatten(labels))
    elif method == 'ratio':
        return np.array([sum(counts) / x for x in counts])
    elif method == 'effective':
        beta = (sum(counts) - 1) / sum(counts)
        return np.array([(1 - beta) / (1 - np.power(beta, c)) for c in counts])
    elif method == 'none':
        return np.array([1 for _ in range(len(unique))])
    else:
        raise ValueError(f'Unknown method for class weight calculation {method}')

