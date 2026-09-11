"""Top-level namespace exposes the measure classes and submodules (GL#38)."""

import inequality


class TestTopLevelClasses:
    def test_classes_importable(self):
        from inequality import (
            Atkinson,
            Gini,
            Gini_Spatial,
            S,
            Schutz,
            Theil,
            TheilD,
            TheilDSim,
        )

        assert Gini is inequality.gini.Gini
        assert Gini_Spatial is inequality.gini.Gini_Spatial
        assert Theil is inequality.theil.Theil
        assert TheilD is inequality.theil.TheilD
        assert TheilDSim is inequality.theil.TheilDSim
        assert Atkinson is inequality.atkinson.Atkinson
        assert Schutz is inequality.schutz.Schutz
        assert S is inequality.polarization.S

    def test_submodules_still_accessible(self):
        for name in (
            "atkinson",
            "gini",
            "pen",
            "polarization",
            "schutz",
            "theil",
            "wolfson",
        ):
            assert hasattr(inequality, name)

    def test_function_names_not_shadowed_by_modules(self):
        # `atkinson` and `pen` remain the submodules, not the functions,
        # so `inequality.atkinson.atkinson` / `inequality.pen.pen` keep working.
        assert callable(inequality.atkinson.atkinson)
        assert callable(inequality.wolfson.wolfson)
        assert callable(inequality.pen.pen)

    def test_dunder_all(self):
        for name in inequality.__all__:
            assert hasattr(inequality, name), name
