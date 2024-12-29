import matplotlib.pyplot as plt

__all__ = ["Schutz"]


class Schutz:
    """The Schutz class calculates measures of inequality in an income
    distribution.

    It calculates the Schutz distance, which is the maximum distance
    between the line of perfect equality and the Lorenz curve.
    Additionally, it computes the intersection point with the line of
    perfect equality where the Schutz distance occurs and the original
    Schutz coefficient.
    See :cite:`schutz1951MeasurementIncome`.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    column_name : str
        The name of the column for which the Schutz coefficient is to
        be calculated.

    Attributes
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    column_name : str
        The name of the column for which the Schutz coefficient is to
        be calculated.
    df_processed : pd.DataFrame
        The processed DataFrame with additional columns.
    distance : float
        The maximum distance between the line of perfect equality and
        the Lorenz curve.
    intersection_point : float
        The x and y coordinate of the intersection point where the
        Schutz distance occurs.
    coefficient : float
        The original Schutz coefficient.

    Examples
    --------
    >>> import pandas as pd
    >>> gdf = pd.DataFrame({
    ...     'NAME': ['A', 'B', 'C', 'D', 'E'],
    ...     'Y': [1000, 2000, 1500, 3000, 2500]
    ... })
    >>> schutz_obj = Schutz(gdf, 'Y')
    >>> print("Schutz Distance:", round(float(schutz_obj.distance),2))
    Schutz Distance: 0.15
    >>> print("Intersection Point:", round(schutz_obj.intersection_point, 1))
    Intersection Point: 0.6
    >>> print("Schutz Coefficient:", round(schutz_obj.coefficient, 1))
    Schutz Coefficient: 7.5
    """

    def __init__(self, df, column_name):
        """
        Initialize the Schutz object, calculate the Schutz distance,
        the intersection point with the line of perfect equality, and
        the original Schutz coefficient.

        Parameters
        ----------
        df: pd.DataFrame
            The input DataFrame containing the data.
        column_name: str
            The name of the column for which the Schutz coefficient is
            to be calculated.
        """
        self.df = df
        self.column_name = column_name
        self.df_processed = self._prepare_dataframe()
        self.distance = self.calculate_schutz_distance()
        self.intersection_point = self.calculate_intersection_point()
        self.coefficient = self.calculate_schutz_coefficient()

    def _prepare_dataframe(self):
        """
        Prepare the DataFrame by sorting and calculating necessary
        columns.

        Returns
        -------
        pd.DataFrame
            The processed DataFrame with additional columns.
        """
        df = (
            self.df[[self.column_name]]
            .sort_values(by=self.column_name)
            .reset_index(drop=True)
        )
        df["unit"] = 1
        df["upct"] = df.unit / df.unit.sum()
        df["ypct"] = df[self.column_name] / df[self.column_name].sum()
        df["ucpct"] = df.upct.cumsum()
        df["ycpct"] = df.ypct.cumsum()
        df["distance"] = df["ucpct"] - df["ycpct"]
        df["slope"] = df.ypct / df.upct
        df["coefficient"] = 10 * (df.slope - 1)
        return df

    def calculate_schutz_distance(self):
        """
        Calculate the Schutz distance, which is the maximum distance
        between the line of perfect equality and the Lorenz curve.

        Returns
        -------
        float
            The maximum distance indicating the level of inequality.
        """
        return self.df_processed["distance"].max()

    def calculate_intersection_point(self):
        """
        Calculate the intersection point of the line of perfect equality
        and the Lorenz curve.

        Returns
        -------
        float
            The x and y coordinate of the intersection point where the
            Schutz distance occurs.
        """
        max_distance_row = self.df_processed[
            self.df_processed["distance"] == self.distance
        ].iloc[0]
        intersection_point = max_distance_row["ucpct"]
        return intersection_point

    def calculate_schutz_coefficient(self):
        """
        Calculate the original Schutz coefficient.

        Returns
        -------
        float
            The Schutz coefficient.
        """
        coefficient = self.df_processed[
            self.df_processed["coefficient"] > 0
        ].coefficient.sum()
        return coefficient

    def plot(
        self,
        xlabel="Cumulative Share of the Population",
        ylabel="Cumulative Share of Income",
        grid=True,
        title=None,
    ):
        """
        Plot the Lorenz curve, the line of perfect equality, and the
        Schutz line.

        The plot shows the Lorenz curve, a 45-degree line representing
        perfect equality, and the Schutz line dropping vertically from
        the intersection point on the line of perfect equality to the
        Lorenz curve.
        """
        plt.figure(figsize=(10, 6))

        # Plot Lorenz curve
        plt.plot(
            [0] + self.df_processed["ucpct"].tolist(),
            [0] + self.df_processed["ycpct"].tolist(),
            label="Lorenz Curve",
            color="blue",
        )

        # Plot 45-degree line of perfect equality
        plt.plot(
            [0, 1],
            [0, 1],
            label="Line of Perfect Equality",
            color="black",
            linestyle="--",
        )

        # Plot Schutz line
        plt.plot(
            [self.intersection_point, self.intersection_point],
            [self.intersection_point, self.intersection_point - self.distance],
            label="Schutz Line",
            color="red",
            linestyle=":",
        )

        # Add labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title is None:
            title = self.column_name
        plt.title(title)
        plt.legend()
        plt.grid(grid)
        plt.show()
