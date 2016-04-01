# -*- coding: utf-8 -*-
__author__ = '''\n'''.join(['Vincent Gauthier <vgauthier@luxbulb.org>'])
__all__ = ['Mfig']
__license__ = "MIT"


import matplotlib
from cycler import cycler
from palettable.colorbrewer.qualitative import Paired_10
import numpy as np


class Mfig(object):
    """
    Example
    -------
    >>> import mplp
    >>> from palettable.colorbrewer.sequential import Blues_8
    >>> mfig = mplp.Mfig('normal', colors=Blues_8)
    >>> fig, ax = mfig.subplots()
    >>> ax.plot(x, y, '-')
    >>> mfig.savefig("plot.pdf")

    Notes
    -----
    .. _The palettable documentation:
    https://jiffyclub.github.io/palettable/

    .. _Colorbrewer:
    http://colorbrewer2.org/
    """

    sizes = {"fontsize": 11, "titlesize": 12, "linewidth": 1.5}

    colors_cyl = Paired_10

    def __init__(self,
                 format=None,
                 scale=0.9,
                 formatting='landscape',
                 colors=None,
                 fontsize=None):
        """Constructor

        Parameters
        ----------
        format: String, Optopnal
            "single": single column paper
            "double": double column paper

        scale : Float, Optional
            default is 1.0

        formatting: String, Optional,
            either 'landscape' or square, default is 'landscape'

        colors : palettable Object, Optional
            the default color palette is the 'Paired_10'
            .. _See the palettable documentation for more information:
            https://jiffyclub.github.io/palettable/

        fontsize : Int, Optional
            fontsize of the tick label, the axis label and the legend label
        """

        if format == "double":
            scale = 0.5
            self.sizes = {"fontsize": 10, "titlesize": 11, "linewidth": 1.3}
        elif format == "single":
            self.sizes = {"fontsize": 15, "titlesize": 14, "linewidth": 1.6}

        # overwrite the fontsize parameter if passed as argument
        if fontsize:
            self.sizes["fontsize"] = fontsize
        self.set_figsize(formatting=formatting, scale=scale)

        if colors:
            self.colors_cyl = colors

        # set matplotlib parameters
        self.set_matplotlib_parameters()

    def set_figsize(self, formatting, scale):
        """
        Parameters
        ----------
        formatting: Sting, Optional
            either 'landscape', 'square', default is landascpe

        scale: float, Optional
            default is 0.9

        .. _copy from Bennett Kanuka blog:
        http://bkanuka.com/articles/native-latex-plots/
        """
        fig_width_pt = 516.0                                # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0 / 72.27                         # Convert pt to inch
        golden_mean = (np.sqrt(5.0) - 1.0) / 2.0            # Aesthetic ratio (you could change this)
        fig_width = fig_width_pt * inches_per_pt * scale    # width in inches
        fig_height = fig_width * golden_mean                # height in inches
        if formatting == 'landscape':
            self.fig_size = (fig_width, fig_height)
        else:
            self.fig_size = (fig_height, fig_height)
        return self.fig_size

    def get_figsize(self):
        """Return the figure dimentions
        Returns
        -------
            (w, h): Tulpe, (width, height)
        """
        return self.fig_size

    def get_sizes(self):
        """Return the fontsize, titlesize, linewidth of the figure resize according to
        the initial format "normal", "wide" or "poster".

        Returns
        -------
            Tulpe: (fontsize, titlesize, linewidth)
        """
        return (self.sizes['fontsize'],
                self.sizes['titlesize'],
                self.sizes['linewidth'])

    def subplots(self, nrows=1, ncols=1, sharex=False, sharey=False):
        """Create a figure with a set of subplots already made.

        Parameters
        ----------
        nrows : int, Optional
            Number of rows of the subplot grid. Defaults to 1.
        ncols : int, Optional
            Number of columns of the subplot grid. Defaults to 1.
        sharex : string or bool, Optional
            If True, the X axis will be shared amongst all subplots. If True and you have
            multiple rows, the x tick labels on all but the last row of plots will have
            visible set to False If a string must be one of “row”, “col”, “all”, or “none”.
            “all” has the same effect as True, “none” has the same effect as False. If
            “row”, each subplot row will share a X axis. If “col”, each subplot column will
            share a X axis and the x tick labels on all but the last row will have visible
            set to False.
        sharey : string or bool, Optional
            If True, the Y axis will be shared amongst all subplots. If True and you have
            multiple columns, the y tick labels on all but the first column of plots will
            have visible set to False If a string must be one of “row”, “col”, “all”, or
            “none”. “all” has the same effect as True, “none” has the same effect as False.
            If “row”, each subplot row will share a Y axis and the y tick labels on all but
            the first column will have visible set to False. If “col”, each subplot column
            will share a Y axis.

        Returns
        -------
            fig, ax return a matplotlib the figure obj, and a axis object
        """
        import matplotlib.pylab as plt

        w, h = self.get_figsize()
        self.fig, self.ax = plt.subplots(nrows,
                                         ncols,
                                         sharex, sharey,
                                         figsize=(w, h))

        return self.fig, self.ax

    def show(self):
        """show the figure in the viewer backend
        """
        self.fig.tight_layout()
        self.fig.show()

    def get_color_cycle(self):
        """Return the color palette
        Returns
        -------
            return a list of colors palette
        """
        return self.colors_cyl.mpl_colors

    def savefig(self, filename, pgf=True):
        """Generate a vector figure compile into a '*.pdf' and a '*.pgf' file

        Parameters
        ----------
        filename : String
            Filename of the picture
        filename : Boolean, optional
            Generate a PGF file
        """

        self.fig.tight_layout()
        self.fig.savefig('{}.pgf'.format(filename), dpi=300, transparent=True)
        if pgf:
            self.fig.savefig('{}.pdf'.format(filename), dpi=300, transparent=True)

    def set_matplotlib_parameters(self):
        """Setup the matplotlib's rc parameters
        """

        (fontsize, titlesize, linewidth) = self.get_sizes()

        params = {
            'text.latex.preamble': [
                r'\usepackage{sfmath}',
                r'\usepackage{amsmath}'
            ],
            'pgf.preamble': [
                r'\usepackage[utf8x]{inputenc}',
                r'\usepackage[T1]{fontenc}',
                r'\usepackage{amsmath}',
                r'\usepackage{sfmath}'
            ]
        }

        matplotlib.rcParams.update(params)
        matplotlib.rcParams['lines.linewidth'] = linewidth
        matplotlib.rcParams['patch.linewidth'] = linewidth
        matplotlib.rcParams['axes.linewidth'] = linewidth
        matplotlib.rcParams['axes.titlesize'] = titlesize
        matplotlib.rcParams['grid.linewidth'] = linewidth
        matplotlib.rcParams['font.size'] = fontsize
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['xtick.major.width'] = linewidth - 0.1
        matplotlib.rcParams['xtick.major.size'] = 4 * linewidth
        matplotlib.rcParams['xtick.minor.width'] = linewidth - 0.1
        matplotlib.rcParams['xtick.minor.size'] = 2 * linewidth
        matplotlib.rcParams['ytick.major.width'] = linewidth - 0.1
        matplotlib.rcParams['ytick.major.size'] = 4 * linewidth
        matplotlib.rcParams['ytick.minor.width'] = linewidth - 0.1
        matplotlib.rcParams['ytick.minor.size'] = 2 * linewidth
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
        # Only on point is allowed in the legend
        matplotlib.rcParams['legend.numpoints'] = 1
        matplotlib.rcParams['font.serif'] = ['Computer Modern Roman']
        matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
        matplotlib.rcParams['font.monospace'] = ['Computer Modern Typewriter']
        # CMAP color palette
        matplotlib.rcParams['image.cmap'] = 'viridis'
        # Line color Pallete
        matplotlib.rcParams['axes.prop_cycle'] = cycler('color', self.colors_cyl.mpl_colors)