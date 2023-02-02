import functools
import napari
import numpy as np
from fibsem_tools.io import read_xarray
from napari.qt.threading import thread_worker
from skimage.transform import resize
from napari.utils.events import Event

from napari_multiscale_rendering_prototype.utils import ChunkCacheManager

import dask.array as da

viewer = napari.Viewer(ndisplay=3)

# TODO checkout from psygnal import debounced

large_image = {
    "container": "s3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5",
    "dataset": "labels/empanada-mito_seg",
    "scale_levels": 4,
}
large_image["real_arrays"] = [
    read_xarray(
        f"{large_image['container']}/{large_image['dataset']}/s{scale}/",
        storage_options={"anon": True},
    )
    for scale in range(large_image["scale_levels"])
]
# Make a fake arrays structure with empty data
large_image["arrays"] = [
    da.ones_like(array) for array in large_image["real_arrays"]
]

large_image["arrays"] = large_image["real_arrays"]

cache_manager = ChunkCacheManager()

print(
    f"Data loaded has: num scales = {len(large_image['arrays'])}, shape: {large_image['arrays'][0].shape}"
)

# This version is based on the data chunk size

# view_interval = ((0, 0, 0), (1536, 1536, 1536))
# view_interval = ((0, 0, 0), (768, 768, 768))
# view_interval = ((0, 0, 0), (384, 384, 384))

# in units of the highest res data, how big are chunks at each scale
# chunk_strides = [[2**scale * el for el in large_image['arrays'][scale].chunksize] for scale in range(large_image['scale_levels'])]
chunk_strides = [
    [2**scale * el for el in large_image["arrays"][scale].data.chunksize]
    for scale in range(large_image["scale_levels"])
]

# Make an interval that is 3x3x3 for the most coarse chunk size
# view_interval = ((0, 0, 0), [3 * el for el in chunk_strides[3]])
view_interval = ((0, 0, 0), (6144, 2048, 4096))


def get_chunk(
    coord, array=None, container=None, dataset=None, chunk_size=(1, 1, 1)
):
    """
    coord - 3D coordinate for the top left corner (in array space)
    array is one of the scales from the multiscale image
    container is the zarr container name (this is used to disambiguate the cache)
    dataset is the group in the zarr (this is used to disambiguate the cache)
    chunk_size is the size of chunk that you want to fetch

    [note there is an issue with mismatched chunk sizes clashing]

    [TODO consider changing coord to a slice tuple]
    """
    real_array = cache_manager.get(container, dataset, coord)
    if real_array is None:
        x, y, z = coord
        real_array = np.asarray(
            array[
                x : (x + chunk_size[0]),
                y : (y + chunk_size[1]),
                z : (z + chunk_size[2]),
            ].compute()
        )
        cache_manager.put(container, dataset, coord, real_array)
    return real_array


def add_subnodes(interval, scale=0, focus=(0, 0, 0), viewer=viewer):
    """
    interval is the region of the image that you want to display this is a tuple of 2 3d tuples, defining the min/max of the region
    scale is the current scale level being fetched (e.g. from the multiscale arrays) 0 is highest resolution
    focus is the point that will be used to determine the chunk that is recursed at the next highest scale
    viewer is a napari viewer that the nodes are added to

    TODO maybe we should smoosh chunks together within the same resolution level
    """

    # Break the interval up according to this scale's chunk stride
    min_coord, max_coord = interval

    intervals = []

    coord = [0, 0, 0]
    chunk_stride = chunk_strides[scale]

    print(
        f"add_subnodes {scale} {chunk_stride} {interval} array shape: {large_image['arrays'][scale].shape}"
    )

    # Create a list of coordinates for all chunks at this scale within the interval
    while coord[2] + chunk_stride[2] <= max_coord[2]:
        coord[1] = 0
        while coord[1] + chunk_stride[1] <= max_coord[1]:
            coord[0] = 0
            while coord[0] + chunk_stride[0] <= max_coord[0]:
                sub_min_coord = [el for el in coord]
                sub_max_coord = [
                    coord[idx] + chunk_stride[idx] for idx in range(len(coord))
                ]

                print(sub_min_coord, sub_max_coord)

                intervals.append((sub_min_coord, sub_max_coord))
                coord[0] += chunk_stride[0]
            coord[1] += chunk_stride[1]
        coord[2] += chunk_stride[2]

    def distancesq(p1, p2):
        return sum([(p1[idx] - p2[idx]) ** 2 for idx in range(len(p1))])

    # get closest interval to focus
    min_idx = 0
    for idx, interval in enumerate(intervals):
        if distancesq(focus, interval[0]) < distancesq(
            focus, intervals[min_idx][0]
        ):
            min_idx = idx

    colormaps = {0: "red", 1: "blue", 2: "green", 3: "yellow"}

    for idx, interval in enumerate(intervals):
        # render remaining, or all if we're on the last scale level
        if idx != min_idx or scale == 0:
            # find position and scale
            min_interval, max_interval = intervals[idx]
            print(
                f"Fetching: {(scale, min_interval[0], min_interval[1], min_interval[2])}"
            )
            dataset = f"{large_image['dataset']}/s{scale}"
            data = get_chunk(
                (
                    int(min_interval[0] / 2**scale),
                    int(min_interval[1] / 2**scale),
                    int(min_interval[2] / 2**scale),
                ),
                array=large_image["arrays"][scale],
                container=large_image["container"],
                dataset=dataset,
                chunk_size=large_image["arrays"][scale].data.chunksize,
            ).transpose()
            node_scale = (
                2**scale,
                2**scale,
                2**scale,
            )
            viewer.add_image(
                data,
                scale=node_scale,
                translate=(min_interval[2], min_interval[1], min_interval[0]),
                name=f"chunk_{(scale, min_interval[0], min_interval[1], min_interval[2])}",
                blending="additive",
                colormap=colormaps[scale],
                opacity=0.8,
                rendering="mip",
            )
            # set data

    # recurse on closest
    if scale > 0:
        print(
            f"Recursive add nodes on {min_idx} {intervals[min_idx]} for scale {scale} to {scale-1}"
        )
        add_subnodes(intervals[min_idx], scale=scale - 1, viewer=viewer)


# use voxel resolution, might be 2x scale

# add_subnodes(((0, 0, 0), arrays[0].shape))
add_subnodes(view_interval, scale=3, viewer=viewer, focus=(2144, 1048, 2096))

# TODO figure out why the focus wont work
# work on poor-mans-octree branch
# migrate this code into it, use the camera management


# napari.run()
