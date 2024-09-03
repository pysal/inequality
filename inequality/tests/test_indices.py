import numpy
import pytest

from inequality._indices import (
    abundance,
    ellison_glaeser_egg,
    ellison_glaeser_egg_pop,
    fractionalization_gs,
    gini_gi,
    gini_gi_m,
    gini_gig,
    herfindahl_hd,
    hoover_hi,
    isolation_ii,
    isolation_isg,
    margalev_md,
    maurel_sedillot_msg,
    maurel_sedillot_msg_pop,
    menhinick_mi,
    modified_segregation_msg,
    polarization,
    segregation_gsg,
    shannon_se,
    similarity_w_wd,
    simpson_sd,
    simpson_so,
    theil_th,
    theil_th_brute,
)

x = numpy.array([[0, 1, 2], [0, 2, 4], [0, 0, 3]])

numpy.random.seed(0)
y = numpy.random.randint(1, 10, size=(4, 3))

numpy.random.seed(0)
tau = numpy.random.uniform(size=(3, 3))
numpy.fill_diagonal(tau, 0.0)
tau = (tau + tau.T) / 2

numpy.random.seed(0)
z = numpy.random.randint(10, 50, size=(3, 4))

numpy.random.seed(0)
v = numpy.random.uniform(0, 1, size=(4,))


class TestAbundance:
    def test_abundance(self):
        known = 2
        with pytest.warning_depr("abundance"):
            observed = abundance(x)
        assert known == observed


class TestMargalevMD:
    def test_margalev_md(self):
        known = 0.40242960438184466
        with pytest.warning_depr("abundance"), pytest.warning_depr("margalev_md"):
            observed = margalev_md(x)
        assert known == pytest.approx(observed)


class TestMenhinickMI:
    def test_menhinick_mi(self):
        known = 0.2886751345948129
        with pytest.warning_depr("abundance"), pytest.warning_depr("menhinick_mi"):
            observed = menhinick_mi(x)
        assert known == pytest.approx(observed)


class TestSimpsonSO:
    def test_simpson_so(self):
        known = 0.5909090909090909
        with pytest.warning_depr("simpson_so"):
            observed = simpson_so(x)
        assert known == pytest.approx(observed)


class TestSimpsonSD:
    def test_simpson_sd(self):
        known = 0.40909090909090906
        with pytest.warning_depr("simpson_so"), pytest.warning_depr("simpson_sd"):
            observed = simpson_sd(x)
        assert known == pytest.approx(observed)


class TestHerfindahlHD:
    def test_herfindahl_hd(self):
        known = 0.625
        with pytest.warning_depr("herfindahl_hd"):
            observed = herfindahl_hd(x)
        assert known == pytest.approx(observed)


class TestTheilTH:
    def test_theil_th(self):
        known = 0.15106563978903298
        with pytest.warning_depr("theil_th"):
            observed = theil_th(x, ridz=True)
        assert known == pytest.approx(observed)

        with (
            pytest.warning_depr("theil_th"),
            pytest.warning_div_zero,
            pytest.warning_invalid("subtract"),
            pytest.warning_invalid("multiply"),
        ):
            observed = theil_th(x, ridz=False)
        assert numpy.isnan(observed)

        # test brute comparison
        with pytest.warning_depr("theil_th_brute"):
            known = theil_th_brute(x, ridz=True)
        with pytest.warning_depr("theil_th"):
            observed = theil_th(x, ridz=True)
        assert known == pytest.approx(observed)

        with (
            pytest.warning_depr("theil_th_brute"),
            pytest.warning_div_zero,
            pytest.warning_invalid("scalar multiply"),
            pytest.warning_invalid("multiply"),
            pytest.warning_invalid("scalar subtract"),
        ):
            observed = theil_th_brute(x, ridz=False)
        assert numpy.isnan(observed)


class TestFractionalizationGS:
    def test_fractionalization_gs(self):
        known = 0.375
        with (
            pytest.warning_depr("herfindahl_hd"),
            pytest.warning_depr("fractionalization_gs"),
        ):
            observed = fractionalization_gs(x)
        assert known == pytest.approx(observed)


class TestPolarization:
    def test_polarization(self):
        with (
            pytest.raises(RuntimeError, match="Not currently implemented."),
            pytest.warning_depr("polarization"),
        ):
            polarization(None)


class TestShannonSE:
    def test_shannon_se(self):
        known = 1.094070862104929
        with pytest.warning_depr("shannon_se"):
            observed = shannon_se(y)
        assert known == pytest.approx(observed)


class TestGiniGI:
    def test_gini_gi(self):
        known = 0.05128205128205132
        with pytest.warning_depr("_gini"), pytest.warning_depr("gini_gi"):
            observed = gini_gi(y)
        assert known == pytest.approx(observed)


class TestGiniGIG:
    def test_gini_gig(self):
        known = numpy.array([0.125, 0.32894737, 0.18181818])
        with pytest.warning_depr("_gini"), pytest.warning_depr("gini_gig"):
            observed = gini_gig(y)
        numpy.testing.assert_array_almost_equal(known, observed)


class TestGiniGIM:
    def test_gini_gi_m(self):
        known = 0.05128205128205132
        with pytest.warning_depr("gini_gi_m"):
            observed = gini_gi_m(y)
        assert known == pytest.approx(observed)


class TestHooverHI:
    def test_hoover_hi(self):
        known = 0.041025641025641046
        with pytest.warning_depr("hoover_hi"):
            observed = hoover_hi(y)
        assert known == pytest.approx(observed)


class TestSimilarityWWD:
    def test_similarity_w_wd(self):
        known = 0.5818596340322582
        with pytest.warning_depr("similarity_w_wd"):
            observed = similarity_w_wd(y, tau)
        assert known == pytest.approx(observed)


class TestSegregationGSG:
    def test_segregation_gsg(self):
        known = numpy.array([0.18292683, 0.24713959, 0.09725159])
        with pytest.warning_depr("segregation_gsg"):
            observed = segregation_gsg(y)
        numpy.testing.assert_array_almost_equal(known, observed)


class TestModifiedSegregationMSG:
    def test_modified_segregation_msg(self):
        known = numpy.array([0.0852071, 0.10224852, 0.0435503])
        with (
            pytest.warning_depr("segregation_gsg"),
            pytest.warning_depr("modified_segregation_msg"),
        ):
            observed = modified_segregation_msg(y)
        numpy.testing.assert_array_almost_equal(known, observed)


class TestIsolationISG:
    def test_isolation_isg(self):
        known = numpy.array([1.0732699, 1.21995329, 1.0227105])
        with pytest.warning_depr("isolation_isg"):
            observed = isolation_isg(y)
        numpy.testing.assert_array_almost_equal(known, observed)


class TestIsolationII:
    def test_isolation_ii(self):
        known = numpy.array([1.1161596, 1.31080357, 1.03432983])
        with pytest.warning_depr("isolation_ii"):
            observed = isolation_ii(y)
        numpy.testing.assert_array_almost_equal(known, observed)


class TestEllisonGlaeserEGG:
    def test_ellison_glaeser_egg(self):
        known = numpy.array([0.0544994, 0.01624183, 0.01014058, 0.02880251])
        with pytest.warning_depr("ellison_glaeser_egg"):
            observed = ellison_glaeser_egg(z)
        numpy.testing.assert_array_almost_equal(known, observed)

        known = numpy.array([-1.0617873, -2.39452501, -1.45991648, -1.11740985])
        with pytest.warning_depr("ellison_glaeser_egg"):
            observed = ellison_glaeser_egg(z, hs=v)
        numpy.testing.assert_array_almost_equal(known, observed)


class TestEllisonGlaeserEGGPop:
    def test_ellison_glaeser_egg_pop(self):
        known = numpy.array([-0.02150826, 0.01329858, -0.03894556])
        with pytest.warning_depr("ellison_glaeser_egg_pop"):
            observed = ellison_glaeser_egg_pop(y)
        numpy.testing.assert_array_almost_equal(known, observed)


class TestMaurelSedillotMSG:
    def test_maurel_sedillot_msg(self):
        known = numpy.array([0.07858256, 0.03597749, 0.03937436, -0.00904911])
        with pytest.warning_depr("maurel_sedillot_msg"):
            observed = maurel_sedillot_msg(z)
        numpy.testing.assert_array_almost_equal(known, observed)

        known = numpy.array([-1.01010171, -2.32421555, -1.38868998, -1.20049894])
        with pytest.warning_depr("maurel_sedillot_msg"):
            observed = maurel_sedillot_msg(z, hs=v.round(3))
        numpy.testing.assert_array_almost_equal(known, observed)


class TestMaurelSedillotMSGPop:
    def test_maurel_sedillot_msg_pop(self):
        known = numpy.array([-0.05503571, 0.04414672, -0.02866628])
        with pytest.warning_depr("maurel_sedillot_msg_pop"):
            observed = maurel_sedillot_msg_pop(y)
        numpy.testing.assert_array_almost_equal(known, observed)
