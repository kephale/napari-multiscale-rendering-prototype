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
from psygnal import debounced
import itertools
from superqt import ensure_main_thread

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
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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


def make_zvalue_images(arrays):
    for array in arrays:
        pass


# TODO try to create a visual debugger with z-values in the multiscale arrays

# large_image = luethi_zenodo_7144919()

cache_manager = ChunkCacheManager()


def get_chunk(
    coord, array=None, container=None, dataset=None
):
    real_array = cache_manager.get(container, dataset, coord)
    if real_array is None:
        z, y, x = coord.astype(np.long)
        real_array = np.asarray(
            array[
                z,
                y : (y + array.data.chunksize[-2]),
                x : (x + array.data.chunksize[-1]),
            ].compute()
        )
        cache_manager.put(container, dataset, coord, real_array)

    return real_array


def interpolated_get_chunk(
    coord, array=None, container=None, dataset=None
):
    """
Coord may be between indices, if so, we need to interpolate
    """
    coord = np.array(coord)
    real_array = cache_manager.get(container, dataset, coord)
    if real_array is None:        
        # If we do not need to interpolate        
        if np.all(coord % 1 == 0):
            real_array = get_chunk(coord, array=array, container=container, dataset=dataset)
        else:
            # Get left and right keys
            lcoord = np.floor(coord)
            rcoord = np.ceil(coord)
            # Handle out of bounds
            try:
                lvalue = get_chunk(lcoord, array=array, container=container, dataset=dataset)
            except:
                lvalue = np.zeros([1] + list(array.data.chunksize[-2:]))
            try:
                rvalue = get_chunk(rcoord, array=array, container=container, dataset=dataset)
            except:
                rvalue = np.zeros([1] + list(array.data.chunksize[-2:]))
            # Linear weight between left/right, assumes parallel
            w = coord[0] - lcoord[0]
            # print(f"interpolated_get_chunk: {lcoord} @ {lvalue.shape}, {rcoord} @ {rvalue.shape}")
            # TODO hardcoded dtype
            real_array = ((1 - w) * lvalue + w * rvalue).astype(np.uint16)
            
        # Save in cache
        cache_manager.put(container, dataset, coord, real_array)
    return real_array


def chunks_for_scale(corner_pixels, array, scale):
    """Scale corner pixels to current scale level

    This function takes corner pixels, and uses metadata from array/scale
    to compute slices to display.
    """

    # TODO all of this needs to be generalized to ND or replaced/merged with volume rendering code

    mins = corner_pixels[0, :] / (2**scale)
    maxs = corner_pixels[1, :] / (2**scale)

    chunk_size = array.data.chunksize

    # z1 = z2 = 150 / (2**scale)

    # TODO kludge for 3D z-only interpolation
    zval = mins[-3]
    
    # Find the extent from the current corner pixels, limit by data shape
    mins = (np.floor(mins / chunk_size) * chunk_size).astype(np.long)
    maxs = np.min(
        (np.ceil(maxs / chunk_size) * chunk_size, np.array(array.shape)),
        axis=0,
    ).astype(np.long)

    mins[-3] = maxs[-3] = zval

    xs = range(mins[-1], maxs[-1], chunk_size[-1])
    ys = range(mins[-2], maxs[-2], chunk_size[-2])
    zs = [zval]

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
        self.data_plane = np.zeros(self.shape[-1 * self.d :], dtype=self.dtype)

    def __getitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        if type(key) is tuple:
            return self.data_plane.__getitem__(tuple(key[-1 * self.d :]))
        else:
            return self.data_plane.__getitem__(key)

    def __setitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol], value
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        if type(key) is tuple:
            return self.data_plane.__setitem__(
                tuple(key[-1 * self.d :]), value
            )
        else:
            return self.data_plane.__setitem__(key, value)


class LinearInterpolatedVirtualData:
    def __init__(self, dtype, shape, source):
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)

        """
If the slice is exact, then return like VirtualData
Otherwise, do a distance-weighted linear interpolation

Out-of-bounds behavior is currently zeros
        """
        self.d = 2

        self.source = source

    def __getitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        if type(key) is tuple:
            skey = self._transform_key(key)
            # If we do not need to interpolate
            if np.all(skey % 1 == 0):
                return self.source.__getitem__(key)
            else:
                # Get left and right keys
                lkey = np.floor(skey)
                rkey = np.ceil(skey)
                # Handle out of bounds
                try:
                    lvalue = self.source.__getitem__(lkey)
                except:
                    lvalue = np.zeros((len(lkey)))
                try:
                    rvalue = self.source.__getitem__(rkey)
                except:
                    rvalue = np.zeros((len(rkey)))
                # Linear weight between left/right
                w = skey[0] - lkey[0]
                return (1 - w) * lvalue + w * rvalue
        else:
            raise Exception("Only tuple indices are supported")

    def __setitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol], value
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        raise Exception("InterpolatedVirtualData.__setitem__ not implemented")


# TODO this class needs to handle interpolation for at least get


# We ware going to use this np array as our "canvas"
# TODO at least get this size from the image
# empty = da.zeros(large_image["arrays"][0].shape[:-1])
# empty = da.zeros(large_image["arrays"][0].shape, chunks=(1, 512, 512))
# empty = zarr.zeros(large_image["arrays"][0].shape, chunks=(1, 512, 512), dtype=np.uint16)

# TODO make a LayerDataProtocol that fakes the size but only uses a 2D space
empty = VirtualData(np.uint16, large_image["arrays"][0].shape)

# TODO let's choose a chunk size that matches the axis we'll be looking at

# What if this was a dask array of zeros like the highest res input array
# empty = da.zeros_like(large_image["arrays"][0])

LOGGER.info(f"canvas {empty.shape} and interpolated")

layer = viewer.add_image(empty, contrast_limits=[20000, 30000])

layer.contrast_limits_range = (0, 1)
layer.contrast_limits = (0, 1)

from itertools import islice

# from multiprocessing import Pool
from threading import Thread


num_threads = 10

def chunk_fetcher(coord, scale, array):
    z, y, x = coord

    # Trigger a fetch of the data
    dataset = f"{large_image['dataset']}/s{scale}"
    LOGGER.info("render_sequence: get_chunk")
    real_array = interpolated_get_chunk(
        (z, y, x),
        array=array,
        container=large_image["container"],
        dataset=dataset,
    )
    # Upscale the data to highest resolution
    upscaled = img_as_uint(
        resize(
            real_array,
            [el * 2**scale for el in real_array.shape],
        )
    )                
    # upscaled = img_as_uint(rescale(real_array, 2**scale))
    
    # Use this to overwrite data and then use a colormap to debug where resolution levels go
    # upscaled = np.ones_like(upscaled) * scale
                
    LOGGER.info(
        f"yielding: {(z * 2**scale, y * 2**scale, x * 2**scale, scale, upscaled.shape)} sample {upscaled[10:20,10]} with sum {upscaled.sum()}"
    )
    # Return upscaled coordinates, the scale, and chunk
    chunk_size = upscaled.shape

    LOGGER.info(
        f"yield will be placed at: {(z * 2**scale, y * 2**scale, x * 2**scale, scale, upscaled.shape)}"
    )

    upscaled_chunk_size = [0, 0]
    upscaled_chunk_size[0] = min(
        large_image["arrays"][0].shape[-2] - y * 2**scale,
        chunk_size[-2],
    )
    upscaled_chunk_size[1] = min(
        large_image["arrays"][0].shape[-1] - x * 2**scale,
        chunk_size[-1],
    )

    # overflow = np.array(large_image["arrays"][0].shape[-2:]) - (np.array((y, x)) + np.array(chunk_size))

    # layer.data[z, y : (y + chunk_size[-2]), x : (x + chunk_size[-1])]

    upscaled = upscaled[
        : upscaled_chunk_size[-2], : upscaled_chunk_size[-1]
    ]

    # if sum(upscaled[10:20,10]) > 0:
    #        import pdb; pdb.set_trace()
    
    # if y == 0 and (x * 2**scale) == 7680:
    #     import pdb; pdb.set_trace()
    
    # TODO pickup here, figure out why this is being placed out of bounds and crop it correctly
    # [z, y : (y + chunk_size[-2]), x : (x + chunk_size[-1])]
    
    return (
        z * 2**scale,
        y * 2**scale,
        x * 2**scale,
        scale,
        upscaled,
        upscaled_chunk_size,
    )

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

        LOGGER.info(
            f"render_sequence: {scale}, {array.shape} fetching {len(chunks_to_fetch)} chunks"
        )

        print(f"Chunks to fetch: {chunks_to_fetch}")

        if num_threads == 1:
            # Single threaded:
            for coord in chunks_to_fetch:
                yield chunk_fetcher(coord, scale, array)

        else:
            # Make a list of num_threads length sublists
            all_job_sets = [
                chunks_to_fetch[i]
                for i in range(0, len(chunks_to_fetch))
            ]

            job_sets = [all_job_sets[idx:idx + num_threads] for idx in range(0, len(all_job_sets), num_threads)]

            def mutable_chunk_fetcher(results, idx, coord, scale, array):
                results[idx] = chunk_fetcher(coord, scale, array)
                # LOGGER.info(f"mutable_get_chunk for scale {scale} updated {coord} @ {idx} as {results[idx].shape}")

            for job_set in job_sets:
                # Evaluate the jobs in parallel
                # chunks = p.map(get_chunk, job_set)
                # We need a mutable result
                results = [None] * len(job_set)
                # Make the threads

                arg_set = [[results, idx] + [args, scale, array] for idx, args in enumerate(job_set)]

                threads = [
                    Thread(
                        target=mutable_chunk_fetcher,
                        args=arg_list
                    )
                    for arg_list in arg_set
                ]
                # Start threads
                for thread in threads:
                    thread.start()

                # Collect
                for thread in threads:
                    thread.join()

                # Yield the chunks that are done
                for idx in range(len(results)):
                    print(f"jobs done: scale {scale} job_idx {idx} job_set {job_set[idx]} with result {results[idx]}")
                    # Get job parameters
                    z, y, x = job_set[idx]

                    chunk_tuple = results[idx]

                    LOGGER.info(f"scale of {scale} upscaled {chunk_tuple[-2].shape} chunksize {chunk_tuple[-1]} at {chunk_tuple[:3]}")
                    
                    # Return upscaled coordinates, the scale, and chunk
                    yield chunk_tuple


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
        LOGGER.info(
            f"Writing chunk with size {chunk_size} to: {(viewer.dims.current_step[0], y, x)}"
        )
        layer.data[
            z, y : (y + chunk_size[-2]), x : (x + chunk_size[-1])
        ] = chunk[: chunk_size[-2], : chunk_size[-1]]
        layer.refresh()

    worker.yielded.connect(on_yield)

    worker.start()



# TODO note that debounced uses threading.Timer
# Connect to camera
viewer.camera.events.connect(
    debounced(
        ensure_main_thread(dims_update_handler),
        timeout=1000,
    )
)

# Connect to dims (to get sliders)
viewer.dims.events.connect(
    debounced(
        ensure_main_thread(dims_update_handler),
        timeout=1000,
    )
)

    

# napari.run()

# TODO there is a problem with interpolation between z-axis
