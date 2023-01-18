import functools
import napari
import numpy as np
from fibsem_tools.io import read_xarray
from napari.qt.threading import thread_worker
from skimage.transform import resize
from napari.utils.events import Event
from itertools import islice
from cachey import Cache

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

# A RenderSequence is a generator that progressively populates data
# in a way that allows it to be rendered when partially loaded.
# Yield after each chunk is fetched
@thread_worker
def render_sequence_singlethread(
    large_image, corner_pixels, cache_manager, chunk_size=(1, 1, 1)
):
    # NOTE this corner_pixels means something else and should be renamed
    # it is further limited to the visible data on the vispy canvas
    arrays = large_image["arrays"]
    container = large_image["container"]

    for scale in reversed(range(4)):
        array = arrays[scale]

        chunks_to_fetch = list(chunks_for_scale(corner_pixels, array, scale))

        dataset = f"{large_image['dataset']}/s{scale}"

        # This is the function that memoizes chunk data
        def get_chunk(coord, chunk_size=(1, 1, 1)):
            real_array = cache_manager.get(container, dataset, coord)
            if real_array:
                return real_array
            else:
                real_array = np.asarray(
                    array[
                        coord[0] : (coord[0] + chunk_size[0]),
                        coord[1] : (coord[1] + chunk_size[1]),
                        coord[2]
                        # coord[2] : (coord[2] + chunk_size[2]),
                    ].compute()
                )
                cache_manager.put(container, dataset, coord, real_array)
            return real_array

        for (x, y, z) in chunks_to_fetch:
            # Trigger a fetch of the data
            real_array = get_chunk((x, y, z))
            # Upscale the data to highest resolution
            upscaled = resize(
                real_array, [el * 2**scale for el in real_array.shape]
            )
            # Return upscaled coordinates, the scale, and chunk
            yield (
                scale,
                (x * 2**scale, y * 2**scale, z * 2**scale),
                upscaled,
            )


@thread_worker
def render_sequence_multithread(arrays, corner_pixels, chunk_size=(1, 1, 1)):
    for scale in reversed(range(4)):
        array = arrays[scale]

        chunks_to_fetch = list(chunks_for_scale(corner_pixels, array, scale))

        # Make a list of num_threads length sublists
        job_sets = [
            chunks_to_fetch[i : i + num_threads]
            for i in range(0, len(chunks_to_fetch), num_threads)
        ]

        # TODO replace the chunk fetcher with a proper implementation
        # TODO use a local zarr disk cache?
        @functools.lru_cache(maxsize=1024)
        def get_chunk(coord, chunk_size=chunk_size):
            real_array = np.asarray(
                array[
                    coord[0] : (coord[0] + chunk_size[0]),
                    coord[1] : (coord[1] + chunk_size[1]),
                    coord[2] : (coord[2] + chunk_size[2]),
                ].compute()
            )
            return real_array

        def mutable_get_chunk(results, idx, array, x, y, z):
            results[idx] = get_chunk((x, y, z))

        for job_set in job_sets:
            # Evaluate the jobs in parallel
            # chunks = p.map(get_chunk, job_set)
            # We need a mutable result
            chunks = [None] * len(job_set)
            # Make the threads
            threads = [
                Thread(
                    target=mutable_get_chunk,
                    args=[chunks, idx, array] + list(args),
                )
                for idx, args in enumerate(job_set)
            ]
            # Start threads
            for thread in threads:
                thread.start()

            # Collect
            for thread in threads:
                thread.join()

            # Yield the chunks that are done
            for idx in range(len(chunks)):
                # Get job parameters
                x, y, z = job_set[idx]
                # This contains the real data
                real_array = chunks[idx]

                # Upscale the data to highest resolution
                upscaled = resize(
                    real_array,
                    [el * 2**scale for el in real_array.shape],
                )

                # Return upscaled coordinates, the scale, and chunk
                yield (
                    scale,
                    (x * 2**scale, y * 2**scale, z),
                    upscaled,
                )


# For debugging purposes
global worker
worker = None


def add_large_image(viewer, large_image, chunk_size=None):
    # We ware going to use this np array as our "canvas"
    arrays = large_image["arrays"]
    empty = np.zeros(arrays[0].shape[::2])
    layer = viewer.add_image(empty)

    layer.contrast_limits_range = (0, 1)
    layer.contrast_limits = (0, 1)

    def dims_update_handler(invar, viewer=viewer):
        global worker

        # This function can be triggered 2 different ways, one way gives us an Event
        if type(invar) is not Event:
            viewer = invar

        # Terminate existing multiscale render pass
        if worker:
            # TODO this might not terminate threads properly
            worker.quit()

        # Find the corners of visible data in the canvas
        corner_pixels = layer.corner_pixels
        canvas_corners = (
            viewer.window.qt_viewer._canvas_corners_in_world.astype(int)
        )

        z = 150

        top_left = (
            int(np.max((corner_pixels[0, 0], canvas_corners[0, 0]))),
            int(np.max((corner_pixels[0, 1], canvas_corners[0, 1]))),
            z,
        )
        bottom_right = (
            int(np.min((corner_pixels[1, 0], canvas_corners[1, 0]))),
            int(np.min((corner_pixels[1, 1], canvas_corners[1, 1]))),
            z,
        )

        corners = np.array([top_left, bottom_right], dtype=int)

        chunk_size_2D = np.array(chunk_size)
        # chunk_size_2D[2] = 1

        # Start a new multiscale render
        # worker = render_sequence_multithread(
        worker = render_sequence_singlethread(
            large_image, corners, cache_manager, chunk_size=chunk_size_2D
        )

        # This will consume our chunks and update the numpy "canvas" and refresh
        def on_yield(response):
            scale, coord, chunk = response
            x, y, z = coord
            chunk_size = chunk.shape
            layer.data[
                y : (y + chunk_size[0]), x : (x + chunk_size[1])
            ] = chunk
            layer.refresh()

        worker.yielded.connect(on_yield)

        worker.start()

    # TODO connect the update to camera/dims changes
    # viewer.dims.events.current_step.connect(dims_update_handler)
    viewer.camera.events.zoom.connect(dims_update_handler)
