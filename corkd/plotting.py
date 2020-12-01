from dataclasses import dataclass
import itertools
import matplotlib
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['figure.dpi'] = 150
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(
    style='whitegrid',
    font_scale=0.8,
    rc={'lines.linewidth': 1.6, 'lines.markersize': 2},
    font='serif',
)
import corkd.kde


@dataclass
class Style:
    contour_levels: int
    line_color: str


default_style = Style(
    contour_levels=4,
    line_color='black',
)


@dataclass
class Axis:
    mpl_ax: object
    style: object

    def add_plot(self, xs, ys):
        self.mpl_ax.plot(xs, ys, color=self.style.line_color)

    def add_contour_plot(self, xs, ys, zs):
        self.mpl_ax.contour(xs, ys, zs.T, levels=self.style.contour_levels, origin='lower', colors=self.style.line_color)

    def unset_shared_y(self, axes_to_remove):
        grouper = self.mpl_ax.get_shared_y_axes()
        for axis in axes_to_remove:
            grouper.remove(axis.mpl_ax)
        self.mpl_ax.yaxis.grid()


class CornerFigure:
    def _get_diagonals(self):
        return [Axis(self._axes[i, i], style=self.style) for i in range(self.ndim)]

    def _get_off_diagonals(self):
        return [Axis(self._axes[i, j], style=self.style) for j in range(self.ndim) for i in range(self.ndim) if (i != j) and (i > j)]

    def clear_upper_off_diag(self):
        for j in range(self.ndim):
            for i in range(self.ndim):
                if (i != j) and (j > i):
                    self._axes[i, j].remove()

    def remove_shared_y_axes_from_diag(self):
        for axis in self.diagonals:
            axis.unset_shared_y(self.off_diagonals)

    def delete_top_left_yticks(self):
        self._axes[0, 0].set_yticks([])

    def __init__(self, ndim, style):
        self.ndim = ndim
        self.style = style
        self._figure, self._axes = plt.subplots(ndim, ndim, sharex='col', sharey='row')
        self.diagonals = self._get_diagonals()
        self.off_diagonals = self._get_off_diagonals()
        self.clear_upper_off_diag()
        self.remove_shared_y_axes_from_diag()
        self.delete_top_left_yticks()

    def set_labels(self, labels):
        for i in range(self.ndim):
            self._axes[i, 0].set_ylabel(labels[i])
            self._axes[-1, i].set_xlabel(labels[i])

    def save_as(self, file_name):
        self._figure.savefig(file_name, bbox_inches='tight')

    def set_size(self, figsize):
        self._figure.set_size_inches(figsize)


class CornerPlot:
    def _check_chains(self, chains):
        if not chains.ndim == 2:
            raise ValueError('\'chains\' must be 2-dimensional')

    def _create_plots(self, densities_1d, densities_2d, labels):
        for density_1d, axis in zip(densities_1d, self.corner_figure.diagonals):
            axis.add_plot(density_1d.grid, density_1d.values)

        for density_2d, axis in zip(densities_2d, self.corner_figure.off_diagonals):
            axis.add_contour_plot(density_2d.grid_1, density_2d.grid_2, density_2d.values)

        if labels is not None:
            self.corner_figure.set_labels(labels)

    def _get_1d_densities(self, chains):
        densities_1d = []
        for i in range(self.ndim):
            densities_1d.append(corkd.kde.Density1D(chains[:, i]))
        return densities_1d

    def _get_all_pairs(self, chains):
        chain_list = [chains[:, i] for i in range(self.ndim)]
        return itertools.combinations(chain_list, 2)

    def _get_2d_densities(self, chains):
        chain_pairs = self._get_all_pairs(chains)
        densities_2d = []
        for chain_pair in chain_pairs:
            densities_2d.append(corkd.kde.Density2D(tuple(chain_pair)))
        return densities_2d

    def __init__(self, chains, labels=None, style=default_style):
        self._check_chains(chains)
        self.ndim = chains.shape[1]
        self.corner_figure = CornerFigure(ndim=self.ndim, style=style)
        densities_1d = self._get_1d_densities(chains)
        densities_2d = self._get_2d_densities(chains)
        self._create_plots(densities_1d, densities_2d, labels)

    def save_as(self, file_name, fig_size=(5.5, 5)):
        self.corner_figure.set_size(fig_size)
        self.corner_figure.save_as(file_name)
