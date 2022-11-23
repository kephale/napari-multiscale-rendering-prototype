import napari
import numpy as np
from fibsem_tools.io import read_xarray
from napari.qt.threading import thread_worker
from skimage.transform import resize

viewer = napari.Viewer()

# TODO fix a lot of hard coding in here
container = "s3://janelia-cosem-datasets/jrc_macrophage-2/jrc_macrophage-2.n5"
dataset = "em/fibsem-uint16"
scale_levels = range(4)
chunk_size = (384, 384, 384)
arrays = [
    read_xarray(
        f"{container}/{dataset}/s{scale}/", storage_options={"anon": True}
    )
    for scale in scale_levels
]

# We ware going to use this np array as our "canvas"
empty = np.zeros((11087, 10000))
layer = viewer.add_image(empty)

layer.contrast_limits_range = (0, 1)
layer.contrast_limits = (0, 1)


# Yield after each chunk is fetched
@thread_worker
def animator(corner_pixels):
    # NOTE this corner_pixels means something else and should be renamed
    # it is further limited to the visible data on the vispy canvas

    for scale in reversed(range(4)):
        array = arrays[scale]

        # Scale corner pixels to current scale level
        y1, x1 = corner_pixels[0, :] / (2**scale)
        y2, x2 = corner_pixels[1, :] / (2**scale)

        # Find the extent from the current corner pixels, limit by data shape
        y1 = int(np.floor(y1 / chunk_size[0]) * chunk_size[0])
        x1 = int(np.floor(x1 / chunk_size[1]) * chunk_size[1])
        y2 = min(
            int(np.ceil(y2 / chunk_size[0]) * chunk_size[0]), array.shape[0]
        )
        x2 = min(
            int(np.ceil(x2 / chunk_size[1]) * chunk_size[1]), array.shape[1]
        )

        xs = range(x1, x2, chunk_size[1])
        ys = range(y1, y2, chunk_size[0])

        for x in xs:
            for y in ys:
                # TODO z is hardcoded now
                z = int(150 / (2**scale))
                # Trigger a fetch of the data
                real_array = np.asarray(
                    array[
                        y : (y + chunk_size[0]), x : (x + chunk_size[1]), z
                    ].compute()
                )
                # Upscale the data to highest resolution
                upscaled = resize(
                    real_array, [el * 2**scale for el in real_array.shape]
                )
                # Return upscaled coordinates, the scale, and chunk
                yield (x * 2**scale, y * 2**scale, z, scale, upscaled)


global worker
worker = None


# Key press will trigger a new multiscale refresh
@viewer.bind_key("k")
def dims_update_handler(viewer):
    global worker

    # Terminate existing multiscale render pass
    if worker:
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
    worker = animator(corners)

    # This will consume our chunks and update the numpy "canvas" and refresh
    def on_yield(coord):
        x, y, z, scale, chunk = coord
        chunk_size = chunk.shape
        layer.data[y : (y + chunk_size[0]), x : (x + chunk_size[1])] = chunk
        layer.refresh()

    worker.yielded.connect(on_yield)

    worker.start()


# TODO connect the update to camera/dims changes
viewer.dims.events.current_step.connect(dims_update_handler)

napari.run()
