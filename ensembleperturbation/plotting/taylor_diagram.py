"""
Taylor diagram (Taylor, 2001) test implementation.

http://www-pcmdi.llnl.gov/about/staff/Taylor/CV/Taylor_diagram_primer.htm
"""

__version__ = 'Time-stamp: <2012-02-17 20:59:35 ycopin>'
__author__ = 'Yannick Copin <yannick.copin@laposte.net>'

# Rev. 1.0  saeed Moghimi  moghimis@gmail.com

from matplotlib import pyplot
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes, grid_finder
import numpy


class TaylorDiagram:
    """Taylor diagram: plot model standard deviation and correlation
    to reference (data) sample in a single-quadrant polar plot, with
    r=stddev and theta=arccos(correlation).
    """

    def __init__(self, refstd, fig=None, rect=111, label='_'):
        """Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using mpl_toolkits.axisartist.floating_axes. refstd is
        the reference standard deviation to be compared to.
        """

        self.refstd = refstd  # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = numpy.concatenate((numpy.arange(10) / 10.0, [0.95, 0.99]))
        tlocs = numpy.arccos(rlocs)  # Conversion to polar angles
        gl1 = grid_finder.FixedLocator(tlocs)  # Positions
        tf1 = grid_finder.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent
        self.smin = 0
        self.smax = 1.5 * self.refstd

        ghelper = floating_axes.GridHelperCurveLinear(
            tr,
            extremes=(0, numpy.pi / 2, self.smin, self.smax),  # 1st quadrant
            grid_locator1=gl1,
            tick_formatter1=tf1,
        )

        if fig is None:
            fig = pyplot.figure()

        ax = floating_axes.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis['top'].set_axis_direction('bottom')  # "Angle axis"
        ax.axis['top'].toggle(ticklabels=True, label=True)
        ax.axis['top'].major_ticklabels.set_axis_direction('top')
        ax.axis['top'].label.set_axis_direction('top')
        ax.axis['top'].label.set_text('Correlation')

        ax.axis['left'].set_axis_direction('bottom')  # "X axis"
        ax.axis['left'].label.set_text('Standard deviation')

        ax.axis['right'].set_axis_direction('top')  # "Y axis"
        ax.axis['right'].label.set_text('Standard deviation')

        ax.axis['right'].toggle(ticklabels=True)
        ax.axis['right'].major_ticklabels.set_axis_direction('left')

        ax.axis['bottom'].set_visible(False)  # Useless

        # Contours along standard deviations
        ax.grid(False)

        self._ax = ax  # Graphical axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates

        # Add reference point and stddev contour
        print('Reference std:', self.refstd)
        # l, = self.ax.plot([0], self.refstd, 'k*',     rem by saeed
        #                  ls='', ms=10, label=label)  rem by saeed
        # t = NP.linspace(0, NP.pi/2)                   rem by saeed
        # r = NP.zeros_like(t) + self.refstd            rem by saeed
        # self.ax.plot(t,r, 'k--', label='_')           rem by saeed

        for ip in range(len(rlocs)):
            x = [self.smin, self.smax]
            y = [numpy.arccos(rlocs[ip]), numpy.arccos(rlocs[ip])]
            self.ax.plot(y, x, 'grey', linewidth=0.25)

        # Collect sample points for latter use (e.g. legend)
        # self.samplePoints = [l]                       rem by saeed
        self.samplePoints = []  # add by saeed

    def add_sample(self, stddev, corrcoef, ref, *args, **kwargs):
        """Add sample (stddev,corrcoeff) to the Taylor diagram. args
        and kwargs are directly propagated to the Figure.plot
        command."""

        (l,) = self.ax.plot(numpy.arccos(corrcoef), stddev, *args, **kwargs)  # (theta,radius)
        self.samplePoints.append(l)

        if ref:
            t = numpy.linspace(0, numpy.pi / 2)  # add by saeed
            r = numpy.zeros_like(t) + stddev  # add by saeed
            self.ax.plot(t, r, 'grey', linewidth=0.25, label='_')  # add by saeed

        return l

    def add_contours(self, levels, data_std, **kwargs):
        """Add constant centered RMS difference contours."""

        rs, ts = numpy.meshgrid(
            numpy.linspace(self.smin, self.smax), numpy.linspace(0, numpy.pi / 2)
        )
        # Compute centered RMS difference
        rms = numpy.sqrt(data_std ** 2 + rs ** 2 - 2 * data_std * rs * numpy.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours
