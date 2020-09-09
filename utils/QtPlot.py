from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class QtPlot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=5, dpi=100, draw_axis=False, size=[0.01, 0.01, 0.98, 0.98]):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor("None")
        self.axes = self.fig.add_axes(size)
        if draw_axis == False:
            self.axes.xaxis.set_visible(False)
            self.axes.yaxis.set_visible(False)

        FigureCanvasQTAgg.__init__(self, self.fig)

        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")

    def scatter(self, x, y, c=None, s=None):
        self.axes.scatter(x, y, c=c, s=s)

    def plot(self, x, y, c=None):
        self.axes.plot(x, y, c=c)

    def contour(self, x, y, z, cmap):
        self.axes.pcolor(x, y, z, cmap=cmap)

    def imshow(self, img, cmap=None, extent=None, interpolation='nearest', alpha=None):
        self.axes.imshow(img, cmap=cmap, extent=extent, interpolation=interpolation, alpha=alpha)

    def show_spectrogram(self, spec):
        self.axes.matshow(spec, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')

    def set_margin(self, m):
        self.axes.margins(m)

    def set_lim(self, x1, x2, y1, y2):
        self.axes.set_xlim([x1, x2])
        self.axes.set_ylim([y1, y2])

    def clear(self):
        self.axes.clear()

    def widthForHeight(self, h):
        print("Test")
        return h

    def heightForWidth(self, w):
        print("Test")
        return w