import numpy as np

__all__ = ["Atkinson"]


def _atkinson(y, epsilon):
    """
    Compute the Atkinson index for a given distribution of income or wealth.

    The Atkinson index is a measure of economic inequality that takes into account
    the social aversion to inequality. It is sensitive to changes in different parts
    of the income distribution depending on the value of the parameter epsilon.

    Parameters
    ----------
    y : array-like
        An array of income or wealth values.
    epsilon : float
        The inequality aversion parameter. Higher values of epsilon give more weight
        to the lower end of the distribution, making the index more sensitive to
        changes in the lower tail.

    Returns
    -------
    float
        The Atkinson index, which ranges from 0 (perfect equality) to 1 (maximum inequality).

    Notes
    -----
    - If epsilon equals 0, the Atkinson index is 0 regardless of the distribution,
      as it implies no aversion to inequality.
    - If epsilon equals 1, the Atkinson index is calculated using the geometric mean.
    - The input array y should contain positive values for a meaningful calculation.

    Example
    -------
    >>> import numpy as np
    >>> incomes = np.array([10, 20, 30, 40, 50])
    >>> float(round(_atkinson(incomes, 0.5), 5))
    0.06315
    >>> float(round(_atkinson(incomes, 1),5))
    0.13161
    """
    y = np.asarray(y)
    if epsilon == 1:
        geom_mean = np.exp(np.mean(np.log(y)))
        return 1 - geom_mean / y.mean()
    else:
        ye = y ** (1 - epsilon)
        ye_bar = ye.mean()
        ye_bar = ye_bar ** (1 / (1 - epsilon))
        return 1 - ye_bar / y.mean()


class Atkinson:
    """
    A class to calculate and store the Atkinson index and the equally distributed equivalent (EDE).

    The Atkinson index is a measure of economic inequality that takes into account the social aversion
    to inequality. The equally distributed equivalent (EDE) represents the level of income that, if
    equally distributed, would give the same level of social welfare as the actual distribution.

    See :cite:`Atkinson_1970_Measurement`.

    Parameters
    ----------
    y : array-like
        An array of income or wealth values.
    epsilon : float
        The inequality aversion parameter. Higher values of epsilon give more weight
        to the lower end of the distribution, making the index more sensitive to
        changes in the lower tail.

    Attributes
    ----------
    y : array-like
        The input array of income or wealth values.
    epsilon : float
        The inequality aversion parameter.
    A : float
        The calculated Atkinson index.
    EDE : float
        The equally distributed equivalent (EDE) of the income or wealth distribution.

    Example
    -------
    >>> incomes = np.array([10, 20, 30, 40, 50])
    >>> atkinson = Atkinson(incomes, 0.5)
    >>> float(round(atkinson.A, 5))
    0.06315
    >>> float(round(atkinson.EDE, 5))
    28.1054
    >>> atkinson = Atkinson(incomes, 1)
    >>> float(round(atkinson.A, 5))
    0.13161
    >>> float(round(atkinson.EDE, 5))
    26.05171
    """

    def __init__(self, y, epsilon):
        self.y = np.asarray(y)
        self.epsilon = epsilon
        self.A = _atkinson(y, epsilon)
        self.EDE = y.mean() * (1 - self.A)


# Example usage
if __name__ == "__main__":
    incomes = np.array([10, 20, 30, 40, 50])

    # Using the _atkinson function
    # Output: 0.06315339222708616
    print(f"_atkinson(incomes, 0.5): {_atkinson(incomes, 0.5)}")
    # Output: 0.1316096384342157
    print(f"_atkinson(incomes, 1): {_atkinson(incomes, 1)}")

    # Using the Atkinson class
    atkinson = Atkinson(incomes, 0.5)
    # Output: 0.06315339222708616
    print(f"Atkinson index (epsilon=0.5): {atkinson.A}")
    # Output: 28.105398233187415
    print(f"EDE (epsilon=0.5): {atkinson.EDE}")

    atkinson = Atkinson(incomes, 1)
    # Output: 0.1316096384342157
    print(f"Atkinson index (epsilon=1): {atkinson.A}")
    # Output: 26.051710846973528
    print(f"EDE (epsilon=1): {atkinson.EDE}")
