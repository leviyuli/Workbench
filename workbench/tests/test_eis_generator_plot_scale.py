"""Nyquist units remain geometrically equal across canvas sizes and views."""
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from workbench.tools.eis_generator.logic import DEFAULT_PARAMETERS
from workbench.tools.eis_generator.tool import EisGeneratorTool


@pytest.mark.parametrize("view", ["Transport detail", "Full frequency range"])
@pytest.mark.parametrize("size", [(5.3, 2.4), (5.3, 4.8), (8.2, 6.8), (15.85, 8.8)])
def test_nyquist_uses_equal_pixels_for_equal_impedance_increments(view, size):
    parameters = {**DEFAULT_PARAMETERS, "ionic_conductivity": 0.01,
                  "electronic_conductivity": 0.02, "view": view}
    figure = EisGeneratorTool().run([], parameters).figures["CL Nyquist"]
    canvas = FigureCanvasAgg(figure)
    figure.set_size_inches(*size)
    canvas.draw()
    for axes in figure.axes:
        origin, x_step, y_step = axes.transData.transform([(0, 0), (0.1, 0), (0, 0.1)])
        x_pixels = np.linalg.norm(x_step - origin)
        y_pixels = np.linalg.norm(y_step - origin)
        assert x_pixels == pytest.approx(y_pixels, rel=1e-10)
    # A subsequent resize must retain the same geometric relationship.
    figure.set_size_inches(size[0] * 0.8, size[1] * 1.1)
    canvas.draw()
    for axes in figure.axes:
        origin, x_step, y_step = axes.transData.transform([(0, 0), (0.1, 0), (0, 0.1)])
        assert np.linalg.norm(x_step - origin) == pytest.approx(
            np.linalg.norm(y_step - origin), rel=1e-10)


def test_redraw_preserves_manual_pan_and_zoom():
    figure = EisGeneratorTool().run([], DEFAULT_PARAMETERS).figures["CL Nyquist"]
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    axes = figure.axes[0]
    axes.set_xlim(0.08, 0.15)
    axes.set_ylim(0.0, 0.04)
    canvas.draw()
    assert axes.get_xlim() == pytest.approx((0.08, 0.15))
    assert axes.get_ylim() == pytest.approx((0.0, 0.04))
