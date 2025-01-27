from functools import wraps

import numpy as np
import pandas as pd


def consistent_input(func):
    @wraps(func)
    def wrapper(data, *args, column=None, **kwargs):
        # If input is a DataFrame, extract the specified column
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError(
                    "For DataFrame input, 'column' argument must be provided."
                )
            data = data[column].values
        # If input is a series, numpy array, or list, no transformation needed
        elif isinstance(data, pd.Series | np.ndarray | list):
            data = np.asarray(data)
        else:
            raise TypeError(
                "Input should be a sequence, numpy array, or pandas DataFrame."
            )

        return func(data, *args, **kwargs)

    return wrapper


# Example function using the decorator
@consistent_input
def compute_mean(data):
    return np.mean(data)


# Usage
# df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
# print(compute_mean(df, column="a"))  # Output: 2.5

# arr = np.array([1, 2, 3, 4])
# print(compute_mean(arr))  # Output: 2.5

# lst = [1, 2, 3, 4]
# print(compute_mean(lst))  # Output: 2.5
