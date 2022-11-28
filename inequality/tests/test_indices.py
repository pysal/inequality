import numpy
import pytest

from inequality._indices import ellison_glaeser_egg  # noqa F401
from inequality._indices import ellison_glaeser_egg_pop  # noqa F401
from inequality._indices import hoover_hi  # noqa F401
from inequality._indices import isolation_ii  # noqa F401
from inequality._indices import isolation_isg  # noqa F401
from inequality._indices import maurel_sedillot_msg  # noqa F401
from inequality._indices import maurel_sedillot_msg_pop  # noqa F401
from inequality._indices import modified_segregation_msg  # noqa F401
from inequality._indices import segregation_gsg  # noqa F401
from inequality._indices import (
    abundance,
    fractionalization_gs,
    gini_gi,
    gini_gi_m,
    gini_gig,
    herfindahl_hd,
    margalev_md,
    menhinick_mi,
    polarization,
    shannon_se,
    simpson_sd,
    simpson_so,
    theil_th,
    theil_th_brute,
)

x = numpy.array([[0, 1, 2], [0, 2, 4], [0, 0, 3]])
numpy.random.seed(0)
y = numpy.random.randint(1, 10, size=(4, 3))


class TestAbundance:
    def test_abundance(self):
        known = 2
        observed = abundance(x)
        assert known == observed


class TestMargalevMD:
    def test_margalev_md(self):
        known = 0.40242960438184466
        observed = margalev_md(x)
        assert known == pytest.approx(observed)


class TestMenhinickMI:
    def test_menhinick_mi(self):
        known = 0.2886751345948129
        observed = menhinick_mi(x)
        assert known == pytest.approx(observed)


class TestSimpsonSO:
    def test_simpson_so(self):
        known = 0.5909090909090909
        observed = simpson_so(x)
        assert known == pytest.approx(observed)


class TestSimpsonSD:
    def test_simpson_sd(self):
        known = 0.40909090909090906
        observed = simpson_sd(x)
        assert known == pytest.approx(observed)


class TestHerfindahlHD:
    def test_herfindahl_hd(self):
        known = 0.625
        observed = herfindahl_hd(x)
        assert known == pytest.approx(observed)


class TestTheilTH:
    def test_theil_th(self):
        known = 0.15106563978903298
        observed = theil_th(x)
        assert known == pytest.approx(observed)

        # test brute comparison
        known = theil_th_brute(x)
        observed = theil_th(x)
        assert known == pytest.approx(observed)


class TestFractionalizationGS:
    def test_fractionalization_gs(self):
        known = 0.375
        observed = fractionalization_gs(x)
        assert known == pytest.approx(observed)


class TestPolarization:
    def test_polarization(self):
        with pytest.raises(RuntimeError, match="Not currently implemented."):
            polarization(None)


class TestShannonSE:
    def test_shannon_se(self):
        known = 1.094070862104929
        observed = shannon_se(y)
        assert known == pytest.approx(observed)


class TestGiniGI:
    def test_gini_gi(self):
        known = 0.05128205128205132
        observed = gini_gi(y)
        assert known == pytest.approx(observed)


class TestGiniGIG:
    def test_gini_gig(self):
        known = numpy.array([0.125, 0.32894737, 0.18181818])
        observed = gini_gig(y)
        numpy.testing.assert_array_almost_equal(known, observed)


class TestGiniGIM:
    def test_gini_gi_m(self):
        known = 0.05128205128205132
        observed = gini_gi_m(y)
        assert known == pytest.approx(observed)
