from __future__ import annotations  # c.f. PEP 563, PEP 649

import multiprocessing as mp
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter

from .utils.logs import logger
from .utils._checks import check_type


class Render(BackgroundPlotter):
    """Render window."""

    def __init__(
        self,
        affine: mp.sharedctypes.SynchronizedArray,
    ):
        super().__init__(auto_update=False, toolbar=False, menu_bar=False)
        check_type(affine, (mp.sharedctypes.SynchronizedArray,), "affine")
        if len(affine) != 16:
            raise ValueError(
                "The affine transformation should be a 16 element array in shared "
                "memory. See affine=mp.Array(c_double, 16)."
            )
        self._affine = affine
        self._points = [
            pv.Sphere(center=center, radius=5)
            for center in [
                (13.68, 0.65, 45.07),
                (28.15, 42.21, -31.08),
                (50.67, -44.29, -28.88),
                (-92.50, 1.43, 14.89),
                (-265, 0, 0)
            ]
        ]
        for point in self._points:
            self.add_mesh(point)
        self.add_callback(self._callback, interval=10)

    def _callback(self):
        affine = np.array(self._affine[:]).reshape(4, 4)
        logger.debug("Applying affine transformation:\n%s\n", affine)
        for point in self._points:
            point.transform(affine)
        self.render()
