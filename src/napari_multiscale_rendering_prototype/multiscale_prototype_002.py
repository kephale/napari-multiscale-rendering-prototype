import functools
import napari
import numpy as np
from fibsem_tools.io import read_xarray
from napari.qt.threading import thread_worker
from skimage.transform import resize
from napari.utils.events import Event
import dask.array as da

from napari_multiscale_rendering_prototype.utils import ChunkCacheManager

global viewer
viewer = napari.Viewer()

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

cache_manager = ChunkCacheManager()


def get_chunk(
    coord, array=None, container=None, dataset=None, chunk_size=(1, 1, 1)
):
    real_array = cache_manager.get(container, dataset, coord)
    if real_array is None:
        x, y, z = coord
        real_array = np.asarray(
            array[
                y : (y + large_image["chunk_size"][0]),
                x : (x + large_image["chunk_size"][1]),
                z
                #                z : (z + large_image["chunk_size"][2]),
            ].compute()
        )
        cache_manager.put(container, dataset, coord, real_array)

    return real_array


def chunks_for_scale(corner_pixels, array, scale):
    # Scale corner pixels to current scale level
    y1, x1 = corner_pixels[0, :] / (2**scale)
    y2, x2 = corner_pixels[1, :] / (2**scale)

    z1 = z2 = 150 / (2**scale)

    # Find the extent from the current corner pixels, limit by data shape
    y1 = int(
        np.floor(y1 / large_image["chunk_size"][0])
        * large_image["chunk_size"][0]
    )
    x1 = int(
        np.floor(x1 / large_image["chunk_size"][1])
        * large_image["chunk_size"][1]
    )
    z1 = int(
        np.floor(z1 / large_image["chunk_size"][2])
        * large_image["chunk_size"][2]
    )
    y2 = min(
        int(
            np.ceil(y2 / large_image["chunk_size"][0])
            * large_image["chunk_size"][0]
        ),
        array.shape[0],
    )
    x2 = min(
        int(
            np.ceil(x2 / large_image["chunk_size"][1])
            * large_image["chunk_size"][1]
        ),
        array.shape[1],
    )
    z2 = min(
        int(
            np.ceil(z2 / large_image["chunk_size"][2])
            * large_image["chunk_size"][2]
        ),
        array.shape[2],
    )

    xs = range(x1, x2, large_image["chunk_size"][1])
    ys = range(y1, y2, large_image["chunk_size"][0])
    zs = range(z1, z2, large_image["chunk_size"][2])

    for x in xs:
        for y in ys:
            for z in zs:
                yield (x, y, z)


# We ware going to use this np array as our "canvas"
# TODO at least get this size from the image
empty = np.zeros(large_image["arrays"][0].shape[:-1])

# What if this was a dask array of zeros like the highest res input array
# empty = da.zeros_like(large_image["arrays"][0])

print(f"canvas {empty.shape}")

layer = viewer.add_image(empty)

layer.contrast_limits_range = (0, 1)
layer.contrast_limits = (0, 1)

from itertools import islice

# from multiprocessing import Pool
from threading import Thread

num_threads = 5

# A RenderSequence is a generator that progressively populates data
# in a way that allows it to be rendered when partially loaded.
# Yield after each chunk is fetched
@thread_worker
def render_sequence(corner_pixels):
    # NOTE this corner_pixels means something else and should be renamed
    # it is further limited to the visible data on the vispy canvas

    for scale in reversed(range(4)):
        array = large_image["arrays"][scale]

        chunks_to_fetch = list(chunks_for_scale(corner_pixels, array, scale))

        if num_threads == 1:
            # Single threaded:
            for (x, y, z) in chunks_to_fetch:
                # Trigger a fetch of the data
                dataset = f"{large_image['dataset']}/s{scale}"
                real_array = get_chunk(
                    (x, y, z),
                    array=array,
                    container=large_image["container"],
                    dataset=dataset,
                    chunk_size=(1, 1, 1),
                )
                # Upscale the data to highest resolution
                upscaled = resize(
                    real_array, [el * 2**scale for el in real_array.shape]
                )
                # Return upscaled coordinates, the scale, and chunk
                yield (x * 2**scale, y * 2**scale, z, scale, upscaled)
        else:
            # Make a list of num_threads length sublists
            job_sets = [
                chunks_to_fetch[i : i + num_threads]
                for i in range(0, len(chunks_to_fetch), num_threads)
            ]

            dataset = f"{large_image['dataset']}/s{scale}"

            def mutable_get_chunk(results, idx, x, y, z):
                results[idx] = get_chunk(
                    (x, y, z),
                    array=array,
                    container=large_image["container"],
                    dataset=dataset,
                )

            for job_set in job_sets:
                # Evaluate the jobs in parallel
                # chunks = p.map(get_chunk, job_set)
                # We need a mutable result
                chunks = [None] * len(job_set)
                # Make the threads
                threads = [
                    Thread(
                        target=mutable_get_chunk,
                        args=[chunks, idx] + list(args),
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
                        x * 2**scale,
                        y * 2**scale,
                        z,
                        scale,
                        upscaled,
                    )


global worker
worker = None


def dims_update_handler(invar):
    global worker, viewer

    # This function can be triggered 2 different ways, one way gives us an Event
    if type(invar) is not Event:
        viewer = invar

    # Terminate existing multiscale render pass
    if worker:
        # TODO this might not terminate threads properly
        worker.quit()

    # Find the corners of visible data in the canvas
    corner_pixels = layer.corner_pixels
    canvas_corners = viewer.window.qt_viewer._canvas_corners_in_world.astype(
        int
    )

    top_left = (
        int(np.max((corner_pixels[0, 0], canvas_corners[0, 0]))),
        int(np.max((corner_pixels[0, 1], canvas_corners[0, 1]))),
    )
    bottom_right = (
        int(np.min((corner_pixels[1, 0], canvas_corners[1, 0]))),
        int(np.min((corner_pixels[1, 1], canvas_corners[1, 1]))),
    )

    corners = np.array([top_left, bottom_right], dtype=int)

    # Start a new multiscale render
    worker = render_sequence(corners)

    # This will consume our chunks and update the numpy "canvas" and refresh
    def on_yield(coord):
        x, y, z, scale, chunk = coord
        chunk_size = chunk.shape
        layer.data[y : (y + chunk_size[0]), x : (x + chunk_size[1])] = chunk
        layer.refresh()

    worker.yielded.connect(on_yield)

    worker.start()


viewer.camera.events.zoom.connect(dims_update_handler)
viewer.camera.events.center.connect(dims_update_handler)


# napari.run()
