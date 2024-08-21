import pandas as pd
import pytest
import numpy as np
from inequality.schutz import (
    Schutz,
)  # Replace 'your_module' with the actual name of the module where Schutz is defined


def test_schutz():
    # Sample DataFrame
    data = np.array([20, 50, 80, 100, 100, 100, 100, 120, 150, 180])
    gdf = pd.DataFrame({"NAME": range(len(data)), "Y": data})

    # Create Schutz object
    schutz_obj = Schutz(gdf, "Y")

    # Assert the Schutz distance
    assert schutz_obj.distance == pytest.approx(0.15, rel=1e-9)

    # Assert the intersection point (x=y)
    assert schutz_obj.intersection_point == pytest.approx(0.3, rel=1e-9)

    # Assert the Schutz coefficient
    assert schutz_obj.coefficient == pytest.approx(15, rel=1e-9)


if __name__ == "__main__":
    pytest.main()
