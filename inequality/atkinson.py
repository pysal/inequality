import numpy as np

__all__ = ["Atkinson", "atkinson"]


def atkinson(y, epsilon):
    """Compute the Atkinson index for a given distribution of income or wealth.

    The Atkinson index is a measure of economic inequality that takes
    into account the social aversion to inequality. It is sensitive to
    changes in different parts of the income distribution depending on
    the value of the parameter epsilon.

    Parameters
    ----------
    y : array-like
        An array of income or wealth values.
    epsilon : float
        The inequality aversion parameter. Higher values of epsilon
        give more weight to the lower end of the distribution, making
        the index more sensitive to changes in the lower tail.

    Returns
    -------
    float
        The Atkinson index, which ranges from 0 (perfect equality) to
        1 (maximum inequality).

    Notes
    -----
    - If epsilon equals 0, the Atkinson index is 0 regardless of the
      distribution, as it implies no aversion to inequality.
    - If epsilon equals 1, the Atkinson index is calculated using the
      geometric mean.
    - The input array y should contain positive values for a
      meaningful calculation.

    Example
    -------
    >>> import numpy as np
    >>> incomes = np.array([10, 20, 30, 40, 50])
    >>> float(round(atkinson(incomes, 0.5), 5))
    0.06315
    >>> float(round(atkinson(incomes, 1), 5))
    0.13161

    """
    y = np.asarray(y)
    if np.any(y <= 0):
        raise ValueError("All values in 'y' must be positive.")
    if epsilon < 0:
        raise ValueError("'epsilon' must be non-negative.")

    mean_y = y.mean()
    if epsilon == 1:
        geom_mean = np.exp(np.mean(np.log(y)))
        return 1 - geom_mean / mean_y
    else:
        ye = y ** (1 - epsilon)
        ye_bar = ye.mean() ** (1 / (1 - epsilon))
        return 1 - ye_bar / mean_y


class Atkinson:
    """A class to calculate and store the Atkinson index and the equally
    distributed equivalent (EDE).

    The Atkinson index is a measure of economic inequality that takes
    into account the social aversion to inequality. The equally
    distributed equivalent (EDE) represents the level of income that,
    if equally distributed, would give the same level of social
    welfare as the actual distribution.

    See: :cite:`Atkinson_1970_Measurement`.

    Parameters
    ----------
    y: array-like
        An array of income or wealth values.
    epsilon: float
        The inequality aversion parameter. Higher values of epsilon
        give more weight to the lower end of the distribution, making
        the index more sensitive to changes in the lower tail.

    Attributes
    ----------
    y: array-like
        The input array of income or wealth values.
    epsilon: float
        The inequality aversion parameter.
    A: float
        The calculated Atkinson index.
    EDE: float
        The equally distributed equivalent (EDE) of the income or
        wealth distribution.

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
        y = np.asarray(y)
        if np.any(y <= 0):
            raise ValueError("All values in 'y' must be positive.")
        if epsilon < 0:
            raise ValueError("'epsilon' must be non-negative.")

        self.y = y
        self.epsilon = epsilon
        self.A = atkinson(self.y, self.epsilon)
        self.EDE = self.y.mean() * (1 - self.A)
