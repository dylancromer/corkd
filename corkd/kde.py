from types import MappingProxyType
from dataclasses import dataclass
import numpy as np
import scipy.stats
import corkd.utils


@dataclass
class Density1D:
    samples: np.ndarray
    kde_kwargs: MappingProxyType
    n_grid_points: int = 50*52

    def _calculate_kde(self, samples):
        return scipy.stats.gaussian_kde(samples, **self.kde_kwargs)

    def _make_grid(self, samples):
        return np.linspace(samples.min(), samples.max(), self.n_grid_points)

    def __post_init__(self):
        self._kde = self._calculate_kde(self.samples)
        self.grid = self._make_grid(self.samples)
        self.values = self._kde(self.grid)


@dataclass
class Density2D:
    samples_pair: tuple
    kde_kwargs: MappingProxyType
    n_grid_points: int = 50

    def _calculate_kde(self, sample_1, sample_2):
        return scipy.stats.gaussian_kde(np.stack((sample_1, sample_2)), **self.kde_kwargs)

    def _get_left_endpoint(self, reference):
        anchor = np.abs(reference)
        return reference - self.EPSILON*anchor

    def _get_left_endpoint(self, reference):
        anchor = np.abs(reference)
        return reference + self.EPSILON*anchor

    def _make_grid(self, sample_1, sample_2):
        grid_1 = np.linspace(sample_1.min(), sample_1.max(), self.n_grid_points)
        grid_2 = np.linspace(sample_2.min(), sample_2.max(), self.n_grid_points+2)
        return grid_1, grid_2

    def __post_init__(self):
        self._kde = self._calculate_kde(*self.samples_pair)
        self.grid_1, self.grid_2 = self._make_grid(*self.samples_pair)
        self.values = self._kde(corkd.utils.cartesian_prod(self.grid_1, self.grid_2).T).reshape(self.n_grid_points, -1)
