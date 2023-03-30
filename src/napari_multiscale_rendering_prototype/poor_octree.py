import threading

import dask.array as da
import napari
import numpy as np
from fibsem_tools.io import read_xarray
from magicgui import magic_factory
from psygnal import debounced
from superqt import ensure_main_thread

from .are_the_chunks_in_view import (
    ChunkCacheManager,
    add_subnodes,
    chunk_centers,
)


@magic_factory(
    call_button="Poor Octree Renderer", large_image={"visible": False}
)
def poor_octree_widget(
    viewer: "napari.viewer.Viewer", large_image: dict, alpha: float = 0.8
):
    cache_manager = ChunkCacheManager(cache_size=6e9)

    # TODO if the lowest scale level of these arrays still exceeds
    # texture memory, # this breaks
    multiscale_arrays = large_image["arrays"]

    # Testing with ones is pretty useful for debugging chunk placement for
    # different scales
    # TODO notice that we're using a ones array for testing instead of
    # real data multiscale_arrays = [da.ones_like(array) for array in
    # multiscale_arrays]

    multiscale_chunk_maps = [
        chunk_centers(array)
        for scale_level, array in enumerate(multiscale_arrays)
    ]

    # view_interval = ((0, 0, 0), multiscale_arrays[0].shape)
    view_slice = [
        slice(0, multiscale_arrays[-1].shape[idx])
        for idx in range(len(multiscale_arrays[-1].shape))
    ]
    # Forcing 3D here
    view_slice = view_slice[-3:]

    viewer.dims.ndisplay = 3

    # Initialize layers
    scale_factors = large_image["scale_factors"]

    # Initialize worker
    worker_map = {}

    viewer_lock = threading.Lock()

    viewer.dims.current_step = (0, 5, 135, 160)

    # Hooks and calls to start rendering
    @viewer.bind_key("k", overwrite=True)
    def camera_response(event):
        print("in camera response", alpha, event)
        add_subnodes(
            event,
            view_slice,
            scale=len(multiscale_arrays) - 1,
            viewer=viewer,
            cache_manager=cache_manager,
            arrays=multiscale_arrays,
            chunk_maps=multiscale_chunk_maps,
            container=large_image["container"],
            dataset=large_image["dataset"],
            scale_factors=scale_factors,
            worker_map=worker_map,
            viewer_lock=viewer_lock,
            alpha=alpha,
        )

    # Trigger the first render pass
    camera_response(None)

    # TODO note that debounced uses threading.Timer
    # Connect to camera
    viewer.camera.events.connect(
        debounced(
            ensure_main_thread(camera_response),
            timeout=1000,
        )
    )

    # Connect to dims (to get sliders)
    viewer.dims.events.connect(
        debounced(
            ensure_main_thread(camera_response),
            timeout=1000,
        )
    )


def read_data(group):
    # container is the path to the data.
    # dataset is the part that is right clicked.

    large_image = {
        "dataset": group,
    }

    paths = []
    for array in group.iter_arrays(recursive=False):
        paths.append(array.zarr_path)

    large_image["container"] = array.zarr_file
    large_image["scale_levels"] = len(paths)

    large_image["paths"] = paths
    large_image["arrays"] = []
    for path in paths:
        result = read_xarray(f"{array.zarr_file}/{path}")
        print(result.shape)

        # TODO extract scale_factors now
        large_image["arrays"].append(
            result.data.rechunk((1, 10, 256, 256)).squeeze()
        )

    # hardcoding this for now...
    large_image["scale_factors"] = [
        (1, 0.1625, 0.1625),
        (1, 0.325, 0.325),
        (1, 0.65, 0.65),
        (1, 1.3, 1.3),
        (1, 2.6, 2.6),
    ]

    return large_image


def add_initial_images(viewer, large_image):
    multiscale_arrays = large_image["arrays"]

    container = large_image["container"]
    dataset = large_image["dataset"]
    scale_factors = large_image["scale_factors"]
    colormaps = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "bop purple"}

    scale = len(multiscale_arrays) - 1
    viewer.add_image(
        da.ones_like(multiscale_arrays[scale], dtype=np.uint16),
        blending="additive",
        scale=scale_factors[scale],
        colormap=colormaps[scale],
        opacity=0.8,
        rendering="mip",
        name=f"{container}/{dataset}/s{scale}",
        contrast_limits=[0, 500],
    )

    for scale in range(len(multiscale_arrays) - 1):

        # TODO Make sure this is still smaller than the array
        scale_shape = np.array(multiscale_arrays[scale + 1].chunksize) * 2

        # TODO this is really not the right thing to do,
        # it is an overallocation
        scale_shape[:-3] = multiscale_arrays[scale + 1].shape[:-3]

        viewer.add_image(
            da.ones(
                scale_shape,
                dtype=np.uint16,
            ),
            blending="additive",
            scale=scale_factors[scale],
            colormap=colormaps[scale],
            opacity=0.8,
            rendering="mip",
            name=f"{container}/{dataset}/s{scale}",
            contrast_limits=[0, 500],
        )


def render_poor_octree(group, viewer):

    large_image = read_data(group)

    add_initial_images(viewer, large_image)

    widget = poor_octree_widget()

    widget(viewer, large_image)

    widget.large_image.bind(large_image)

    viewer.window.add_dock_widget(widget, name="Poor Octree Renderer")
