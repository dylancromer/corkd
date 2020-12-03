import pytest
import numpy as np
import corkd


def describe_CornerPlot():

    @pytest.fixture
    def chains():
        return np.loadtxt('test-chains.txt.gz', max_rows=101)

    @pytest.fixture
    def labels():
        return [r'$a_\mathrm{SZ}$', r'$c$', r'$\alpha$', r'$\beta$']

    def it_creates_a_cornerplot_of_kd_estimates(chains, labels):
        cornerplot = corkd.CornerPlot(chains, labels=labels)
        cornerplot.save_as('test_cornerplot.pdf')

    def it_only_allows_2d_chains():
        with pytest.raises(ValueError) as excinfo:
            corkd.CornerPlot(np.ones((2)))
        assert '\'chains\' must be 2-dimensional' in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            corkd.CornerPlot(np.ones((2, 2, 2)))
        assert '\'chains\' must be 2-dimensional' in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            corkd.CornerPlot(np.ones(10*(2,)))
        assert '\'chains\' must be 2-dimensional' in str(excinfo.value)

    @pytest.fixture
    def chains_2():
        return np.loadtxt('test-chains-2.txt.gz', max_rows=101)

    @pytest.fixture
    def labels_2():
        return [r'$a_\mathrm{SZ}$', r'$a_\mathrm{2h}$', r'$c$', r'$\alpha$', r'$\beta$']

    def it_creates_a_cornerplot_of_kd_estimates_2(chains_2, labels_2):
        cornerplot = corkd.CornerPlot(chains_2, labels=labels_2)
        cornerplot.save_as('test_cornerplot_2.pdf')
