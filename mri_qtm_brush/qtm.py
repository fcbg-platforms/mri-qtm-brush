from __future__ import annotations

import asyncio
import multiprocessing as mp

import numpy as np
import qtm_rt

from ._transform import fit_matched_points, quat_to_affine
from .utils._checks import check_type
from .utils.logs import logger


def on_packet_callback(packet):
    """Callback function that is called every time a data packet arrives from QTM."""  # noqa: D401, E501
    global affine_array
    global markers_old

    _, points = packet.get_3d_markers()
    markers = np.array(points)
    if markers_old is not None and markers.shape == markers_old.shape:
        quat, _ = fit_matched_points(markers_old, markers, scale=False)
        with affine_array.get_lock():
            trans = quat_to_affine(quat)
            affine_array[:] = trans.flatten()
        logger.debug("Affine transformation:\n%s\n", trans)
    markers_old = markers


async def _setup(ip: str, version: str):
    """Connect and stream from QTM."""
    logger.debug("Connecting to %s with version %s.", ip, version)
    connection = await qtm_rt.connect(ip, version=version)
    if connection is not None:
        logger.debug("Connected.")
        await connection.stream_frames(components=["3d"], on_packet=on_packet_callback)


def setup(
    affine: mp.sharedctypes.SynchronizedArray,
    ip: str = "127.0.0.1",
    version: str = "1.8",
):
    """Connect and stream from QTM.

    This function should be called from by a separate process.

    Parameters
    ----------
    affine : Array
        Shared variable (:class:`~c_types.c_double`) in which the affine transformation
        is stored.
    ip : str
        IP from which Qualisys Track Manager can be reached.
        ``"127.0.0.1"`` for localhost.
    version : str
        Version of the QTM real-time protocol.
    """
    global affine_array
    global markers_old

    # prepare variable
    check_type(affine, (mp.sharedctypes.SynchronizedArray,), "affine")
    if len(affine) != 16:
        raise ValueError(
            "The affine transformation should be a 16 element array in shared memory. "
            "See affine=mp.Array(c_double, 16)."
        )
    check_type(ip, (str,), "ip")
    check_type(version, (str,), "version")
    affine_array = affine
    markers_old = None

    # prepare asyncio event loop in a separate thread
    asyncio.get_event_loop().create_task(_setup(ip, version))
    asyncio.get_event_loop().run_forever()
