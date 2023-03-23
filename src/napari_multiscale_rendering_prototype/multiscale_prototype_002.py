import logging
import functools
import napari
import numpy as np
from fibsem_tools import read_xarray
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from napari.qt.threading import thread_worker
from skimage.transform import resize, rescale
from skimage.util import img_as_uint
from napari.utils.events import Event
import dask.array as da
import sys
import zarr

import itertools

from napari.utils import config

from napari_multiscale_rendering_prototype.utils import ChunkCacheManager

from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

from napari.layers._data_protocols import LayerDataProtocol, Index

global viewer
viewer = napari.Viewer()

# config.async_loading = True

LOGGER = logging.getLogger("tiled_rendering_2D")
LOGGER.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
streamHandler.setFormatter(formatter)
LOGGER.addHandler(streamHandler)


large_image = {
    "container": "s3://janelia-cosem-datasets/jrc_macrophage-2/jrc_macrophage-2.n5",
    "dataset": "em/fibsem-uint16",
    "scale_levels": 4,
    "chunk_size": (384, 384, 384),
}
large_image["arrays"] = [
    read_xarray(
        f"{large_image['container']}/{large_image['dataset']}/s{scale}/",
        storage_options={"anon": True},
    )
    for scale in range(large_image["scale_levels"])
]

def luethi_zenodo_7144919():
    import os
    import pooch

    # Downloaded from https://zenodo.org/record/7144919#.Y-OvqhPMI0R
    # TODO use pooch to fetch from zenodo
    # zip_path = pooch.retrieve(
    #     url="https://zenodo.org/record/7144919#.Y-OvqhPMI0R",
    #     known_hash=None,# Update hash
    # )
    dest_dir = pooch.retrieve(
        url="https://zenodo.org/record/7144919/files/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip?download=1",
        known_hash="e6773fc97dcf3689e2f42e6504e0d4f4d0845c329dfbdfe92f61c2f3f1a4d55d",
        processor=pooch.Unzip(),
    )
    local_container = os.path.split(dest_dir[0])[0]
    LOGGER.info(local_container)
    store = parse_url(local_container, mode="r").store
    reader = Reader(parse_url(local_container))
    nodes = list(reader())
    image_node = nodes[0]
    dask_data = image_node.data

    large_image = {
        "container": local_container,
        "dataset": "B/03/0",
        "scale_levels": 5,
        "scale_factors": [
            (1, 0.1625, 0.1625),
            (1, 0.325, 0.325),
            (1, 0.65, 0.65),
            (1, 1.3, 1.3),
            (1, 2.6, 2.6),
        ],
        "chunk_size": (1, 10, 256, 256)
    }
    large_image["arrays"] = []
    for scale in range(large_image["scale_levels"]):
        array = dask_data[scale]

        # TODO extract scale_factors now

        # large_image["arrays"].append(result.data.rechunk((3, 10, 256, 256)))
        large_image["arrays"].append(
            array.rechunk((1, 10, 256, 256)).squeeze()
            # result.data[2, :, :, :].rechunk((10, 256, 256)).squeeze()
        )
    return large_image

# large_image = luethi_zenodo_7144919()

cache_manager = ChunkCacheManager()


def get_chunk(
    coord, array=None, container=None, dataset=None, chunk_size=(1, 1, 1)
):
    real_array = cache_manager.get(container, dataset, coord)
    if real_array is None:
        x, y, z = coord
        real_array = np.asarray(
            array[
                z,
                y : (y + array.data.chunksize[-2]),
                x : (x + array.data.chunksize[-1]),
            ].compute()
        )
        cache_manager.put(container, dataset, coord, real_array)

    return real_array


def chunks_for_scale(corner_pixels, array, scale):
    """ Scale corner pixels to current scale level

    This function takes corner pixels, and uses metadata from array/scale
    to compute slices to display.
    """

    # TODO all of this needs to be generalized to ND or replaced/merged with volume rendering code
    
    mins = corner_pixels[0, :] / (2**scale)
    maxs = corner_pixels[1, :] / (2**scale)

    chunk_size = array.data.chunksize
    
    # z1 = z2 = 150 / (2**scale)

    # Find the extent from the current corner pixels, limit by data shape
    mins = (np.floor(mins / chunk_size) * chunk_size).astype(np.long)
    maxs = np.min((np.ceil(maxs / chunk_size) * chunk_size, np.array(array.shape)), axis=0).astype(np.long)

    xs = range(mins[-1], maxs[-1], chunk_size[-1])
    ys = range(mins[-2], maxs[-2], chunk_size[-2])
    zs = range(mins[-3], maxs[-3], chunk_size[-3])
    
    # zs = [z1]
    # TODO kludge
    
    for x in xs:
        for y in ys:
            for z in zs:
                yield (z, y, x)


class VirtualData:
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)

        """
We could call these MarginArrays because they operate on Marginals of dimensions
However, they are sided.
SideMargin

We want to be able to reorient our internal data plane.
        no we should be immutable

        
        """
        self.d = 2
        
        # TODO: I don't like that this is making a choice of slicing axis
        self.data_plane = np.zeros(self.shape[-1 * self.d:], dtype=self.dtype)

    def __getitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        if type(key) is tuple:
            return self.data_plane.__getitem__(tuple(key[-1 * self.d:]))
        else:
            return self.data_plane.__getitem__(key)

    def __setitem__(
            self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol],
            value
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        if type(key) is tuple:
            return self.data_plane.__setitem__(tuple(key[-1 * self.d:]), value)
        else:
            return self.data_plane.__setitem__(key, value)
        


                
# We ware going to use this np array as our "canvas"
# TODO at least get this size from the image
#empty = da.zeros(large_image["arrays"][0].shape[:-1])
# empty = da.zeros(large_image["arrays"][0].shape, chunks=(1, 512, 512))
# empty = zarr.zeros(large_image["arrays"][0].shape, chunks=(1, 512, 512), dtype=np.uint16)

# TODO make a LayerDataProtocol that fakes the size but only uses a 2D space
empty = VirtualData(np.uint16, large_image["arrays"][0].shape)

# TODO let's choose a chunk size that matches the axis we'll be looking at

# What if this was a dask array of zeros like the highest res input array
# empty = da.zeros_like(large_image["arrays"][0])

LOGGER.info(f"canvas {empty.shape}")

layer = viewer.add_image(empty)

layer.contrast_limits_range = (0, 1)
layer.contrast_limits = (0, 1)

from itertools import islice

# from multiprocessing import Pool
from threading import Thread

num_threads = 1

# A RenderSequence is a generator that progressively populates data
# in a way that allows it to be rendered when partially loaded.
# Yield after each chunk is fetched
@thread_worker
def render_sequence(corner_pixels):
    # NOTE this corner_pixels means something else and should be renamed
    # it is further limited to the visible data on the vispy canvas

    LOGGER.info(f"render_sequence: inside with corner pixels {corner_pixels}")

    # TODO scale is hardcoded here
    for scale in reversed(range(4)):
        array = large_image["arrays"][scale]        
        
        chunks_to_fetch = list(chunks_for_scale(corner_pixels, array, scale))

        LOGGER.info(f"render_sequence: {scale}, {array.shape} fetching {len(chunks_to_fetch)} chunks")        

        if num_threads == 1:
            # Single threaded:
            for (z, y, x) in chunks_to_fetch:
                # Trigger a fetch of the data
                dataset = f"{large_image['dataset']}/s{scale}"
                LOGGER.info("render_sequence: get_chunk")
                real_array = get_chunk(
                    (z, y, x),
                    array=array,
                    container=large_image["container"],
                    dataset=dataset,
                    chunk_size=(1, 1, 1),
                )
                # Upscale the data to highest resolution
                # upscaled = resize(
                #     real_array, [el * 2**scale for el in real_array.shape]
                # )
                upscaled = img_as_uint(rescale(real_array, 2**scale))
                
                LOGGER.info(f"yielding: {(z * 2**scale, y * 2**scale, x * 2**scale, scale, upscaled.shape)} sample {upscaled[10:20,10]} with sum {upscaled.sum()}")
                # Return upscaled coordinates, the scale, and chunk
                chunk_size = upscaled.shape

                LOGGER.info(f"yield will be placed at: {(z * 2**scale, y * 2**scale, x * 2**scale, scale, upscaled.shape)}")
                
                upscaled_chunk_size = [0, 0]
                upscaled_chunk_size[0] = min(large_image["arrays"][0].shape[-2] - y * 2**scale, chunk_size[-2])
                upscaled_chunk_size[1] = min(large_image["arrays"][0].shape[-1] - x * 2**scale, chunk_size[-1])
                
                # overflow = np.array(large_image["arrays"][0].shape[-2:]) - (np.array((y, x)) + np.array(chunk_size))
                
                # layer.data[z, y : (y + chunk_size[-2]), x : (x + chunk_size[-1])]
                
                upscaled = upscaled[:upscaled_chunk_size[-2], :upscaled_chunk_size[-1]]

                # if sum(upscaled[10:20,10]) > 0:
                #        import pdb; pdb.set_trace()
                
                # if y == 0 and (x * 2**scale) == 7680:
                #     import pdb; pdb.set_trace()
                
                # TODO pickup here, figure out why this is being placed out of bounds and crop it correctly
                # [z, y : (y + chunk_size[-2]), x : (x + chunk_size[-1])]

                yield (z * 2**scale, y * 2**scale, x * 2**scale, scale, upscaled, upscaled_chunk_size)
        # TODO all this needs to be updated for (xyz) -> (zyx) change
        # else:
        #     # Make a list of num_threads length sublists
        #     job_sets = [
        #         chunks_to_fetch[i : i + num_threads]
        #         for i in range(0, len(chunks_to_fetch), num_threads)
        #     ]

        #     dataset = f"{large_image['dataset']}/s{scale}"

        #     def mutable_get_chunk(results, idx, x, y, z):
        #         results[idx] = get_chunk(
        #             (x, y, z),
        #             array=array,
        #             container=large_image["container"],
        #             dataset=dataset,
        #         )

        #     for job_set in job_sets:
        #         # Evaluate the jobs in parallel
        #         # chunks = p.map(get_chunk, job_set)
        #         # We need a mutable result
        #         chunks = [None] * len(job_set)
        #         # Make the threads
        #         threads = [
        #             Thread(
        #                 target=mutable_get_chunk,
        #                 args=[chunks, idx] + list(args),
        #             )
        #             for idx, args in enumerate(job_set)
        #         ]
        #         # Start threads
        #         for thread in threads:
        #             thread.start()

        #         # Collect
        #         for thread in threads:
        #             thread.join()

        #         # Yield the chunks that are done
        #         for idx in range(len(chunks)):
        #             # Get job parameters
        #             x, y, z = job_set[idx]
        #             # This contains the real data
        #             real_array = chunks[idx]
        #             # Upscale the data to highest resolution
        #             upscaled = resize(
        #                 real_array,
        #                 [el * 2**scale for el in real_array.shape],
        #             )
        #             # Return upscaled coordinates, the scale, and chunk
        #             yield (
        #                 x * 2**scale,
        #                 y * 2**scale,
        #                 z,
        #                 scale,
        #                 upscaled,
        #             )


global worker
worker = None


def dims_update_handler(invar):
    global worker, viewer

    LOGGER.info("dims_update_handler")
    
    # This function can be triggered 2 different ways, one way gives us an Event
    if type(invar) is not Event:
        viewer = invar

    # Terminate existing multiscale render pass
    if worker:
        # TODO this might not terminate threads properly
        worker.quit()

    # Find the corners of visible data in the canvas
    corner_pixels = viewer.layers[0].corner_pixels
    canvas_corners = viewer.window.qt_viewer._canvas_corners_in_world.astype(
        int
    )

    top_left = np.max((corner_pixels, canvas_corners), axis=0)[0, :]
    bottom_right = np.min((corner_pixels, canvas_corners), axis=0)[1, :]

    # TODO Image.corner_pixels behaves oddly maybe b/c VirtualData
    if bottom_right.shape[0] > 2:
        bottom_right[0] = canvas_corners[1, 0]

    corners = np.array([top_left, bottom_right], dtype=int)

    LOGGER.info("dims_update_handler: start render_sequence")
    
    # Start a new multiscale render
    worker = render_sequence(corners)

    LOGGER.info("dims_update_handler: started render_sequence")

    # This will consume our chunks and update the numpy "canvas" and refresh
    def on_yield(coord):
        layer = viewer.layers[0]
        z, y, x, scale, chunk, chunk_size = coord
        # chunk_size = chunk.shape
        LOGGER.info(f"Writing chunk with size {chunk_size} to: {(viewer.dims.current_step[0], y, x)}")
        layer.data[z, y : (y + chunk_size[-2]), x : (x + chunk_size[-1])] = chunk[:chunk_size[-2], :chunk_size[-1]]
        layer.refresh()

    worker.yielded.connect(on_yield)

    worker.start()


viewer.camera.events.zoom.connect(dims_update_handler)
viewer.camera.events.center.connect(dims_update_handler)


# napari.run()

