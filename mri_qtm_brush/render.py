from __future__ import annotations  # c.f. PEP 563, PEP 649

import multiprocessing as mp
import numpy as np

from pyvista import get_reader
from pyvistaqt import BackgroundPlotter

from .utils.logs import logger
from .utils._checks import check_type


class Render(BackgroundPlotter):
    def __init__(self, affine: mp.sharedctypes.SynchronizedArray):
        """Create rendering plotter."""
        super().__init__()
        check_type(affine, (mp.sharedctypes.SynchronizedArray,), "affine")
        if len(affine) != 16:
            raise ValueError(
                "The affine transformation should be a 16 element array in shared "
                "memory. See affine=mp.Array(c_double, 16)."
            )
        self._affine = affine
        reader = get_reader(
            "C:/Users/mathieu.scheltienne/Documents/qtm-ekansh/stl/brush-003.obj"
        )
        self._mesh_brush = reader.read()
        self.add_mesh(self._mesh_brush)
        self.add_callback(self._callback, interval=100)

    def _callback(self):
        affine = np.array(self._affine[:]).reshape(4, 4)
        logger.debug("Applying affine transformation:\n%s\n", affine)
        self._mesh_brush.transform(affine)
