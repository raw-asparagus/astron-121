"""plotting subpackage for Lab 1."""

from ugradio_lab1.plotting.axes_plots import *  # noqa: F401,F403
from ugradio_lab1.plotting.figure_builders import *  # noqa: F401,F403

# Merge __all__ from both submodules.
from ugradio_lab1.plotting.axes_plots import __all__ as _axes_all
from ugradio_lab1.plotting.figure_builders import __all__ as _builders_all

__all__ = sorted({*_axes_all, *_builders_all})
