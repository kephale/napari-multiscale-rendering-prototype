import functools
import napari
import numpy as np
from fibsem_tools.io import read_xarray
from napari.qt.threading import thread_worker
from skimage.transform import resize
from napari.utils.events import Event
from itertools import islice
from cachey import Cache
import dask.array as da

# from multiprocessing import Pool
from threading import Thread

num_threads = 5

# Given an interval, return the keys for the chunkstore
def chunks_for_scale(corner_pixels, array, scale, chunk_size=(1, 1, 1)):
    # Scale corner pixels to current scale level
    y1, x1, z1 = corner_pixels[0, :] / (2**scale)
    y2, x2, z2 = corner_pixels[1, :] / (2**scale)

    # Find the extent from the current corner pixels, limit by data shape
    z1 = int(np.floor(z1 / chunk_size[0]) * chunk_size[0])
    y1 = int(np.floor(y1 / chunk_size[1]) * chunk_size[1])
    x1 = int(np.floor(x1 / chunk_size[2]) * chunk_size[2])
    z2 = min(int(np.ceil(z2 / chunk_size[0]) * chunk_size[0]), array.shape[0])
    y2 = min(int(np.ceil(y2 / chunk_size[1]) * chunk_size[1]), array.shape[1])
    x2 = min(int(np.ceil(x2 / chunk_size[2]) * chunk_size[2]), array.shape[2])

    xs = range(x1, x2, chunk_size[2])
    ys = range(y1, y2, chunk_size[1])
    zs = range(z1, z2, chunk_size[0])

    for x in xs:
        for y in ys:
            for z in zs:
                yield (x, y, z)


# A ChunkCacheManager manages multiple chunk caches
class ChunkCacheManager:
    def __init__(self):
        self.c = Cache(4e9, 0)

    def put(self, container, dataset, key, value):
        """Associate value with key in the given container.
        Container might be a zarr/dataset, key is the index of a chunk, and
        value is the chunk itself."""
        k = self.get_container_key(container, dataset, key)
        self.c.put(k, value, cost=1)

    def get_container_key(self, container, dataset, key):
        return f"{container}/{dataset}@{key}"

    def get(self, container, dataset, key):
        return self.c.get(self.get_container_key(container, dataset, key))


cache_manager = ChunkCacheManager()

# A RenderArray is a ndarray that has a shape like its original source
# but only the rendered region of the array has valid pixel data.
class RenderArray:
    def __init__(self, large_image):
        self.shape = large_image["arrays"][0].shape

        self.empty = da.zeros(large_image["arrays"][0].shape[:-1])

    # TODO this could have the array persistence as well
    # TODO add render_sequence
    # TODO add get_chunk
    # TODO add chunks_for_scale
    # TODO add update handler
