import functools
import napari
import numpy as np
from fibsem_tools.io import read_xarray
from napari.qt.threading import thread_worker
from skimage.transform import resize
from napari.utils.events import Event

global viewer
viewer = napari.Viewer(ndisplay=3)

# TODO fix a lot of hard coding in here
container = "s3://janelia-cosem-datasets/jrc_macrophage-2/jrc_macrophage-2.n5"
dataset = "em/fibsem-uint16"
scale_levels = range(4)
chunk_size = (384, 384, 384)
arrays = [
    read_xarray(f"{container}/{dataset}/s{scale}/", storage_options={"anon": True})
    for scale in scale_levels
]

print(f"Data loaded has: num scales = {len(arrays)}, shape: {arrays[0].shape}")

# This version is based on the data chunk size

# view_interval = ((0, 0, 0), (1536, 1536, 1536))
view_interval = ((0, 0, 0), (768, 768, 768))
# view_interval = ((0, 0, 0), (384, 384, 384))

# open zarr globally
zarrs = {}

import zarr
import sys

local_path = "/Volumes/kish@CZI.T7/jrc_macrophage-2.n5"

for scale, array in reversed(list(enumerate(arrays))):
    group_name = f"{dataset}/s{scale}/"
    # use a directory store
    store = zarr.DirectoryStore(local_path, dimension_separator="/")

    print(f"copying {scale} {group_name}")
    z = zarr.open(
        store=store,
        overwrite=True,
        shape=array.shape,
        dtype=array.dtype,
        path=group_name,
    )

    #    z[:] = array[:]

    zarrs[scale] = z


# This is the chunk store, this call is expensive
# @functools.lru_cache(maxsize=1024)
def get_chunk(scale, x, y, z):
    array = arrays[scale]
    print(f"get_chunk( {scale}, {x}, {y}, {z} ), array shape: {array.shape}")

    real_array = np.asarray(
        array[
            y : (y + chunk_size[0]), x : (x + chunk_size[1]), z : (z + chunk_size[2])
        ].compute()
    )

    zarrfile = zarrs[scale]
    zarrfile[
        y : (y + chunk_size[0]), x : (x + chunk_size[1]), z : (z + chunk_size[2])
    ] = real_array

    return real_array


# Given an interval, return the keys for the chunkstore
def chunks_for_scale(corner_pixels, array, scale):
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
                yield (scale, x, y, z)


def add_subnodes(interval, scale=0, focus=(0, 0, 0)):

    min_coord, max_coord = interval
    mid_coord = (
        int(min_coord[0] + (max_coord[0] - min_coord[0]) / 2),
        int(min_coord[1] + (max_coord[1] - min_coord[1]) / 2),
        int(min_coord[2] + (max_coord[2] - min_coord[2]) / 2),
    )

    data_scale = scale if scale < 4 else 3

    intervals = []
    # chop interval
    for qx in range(2):
        for qy in range(2):
            for qz in range(2):
                sx, ex = (
                    (min_coord[0], mid_coord[0])
                    if qx == 0
                    else (
                        mid_coord[0],
                        max_coord[0],
                    )
                )
                sy, ey = (
                    (min_coord[1], mid_coord[1])
                    if qy == 0
                    else (
                        mid_coord[1],
                        max_coord[1],
                    )
                )
                sz, ez = (
                    (min_coord[2], mid_coord[2])
                    if qz == 0
                    else (
                        mid_coord[2],
                        max_coord[2],
                    )
                )
                sub_min_coord = (sx, sy, sz)
                sub_max_coord = (ex, ey, ez)
                intervals.append((sub_min_coord, sub_max_coord))

    def distance(p1, p2):
        return sum([(p1[idx] - p2[idx]) ** 2 for idx in range(len(p1))]) ** 0.5

    # get closest interval to focus
    min_idx = 0
    for idx, interval in enumerate(intervals):
        if distance(focus, interval[0]) < distance(focus, intervals[min_idx][0]):
            min_idx = idx


    for idx, interval in enumerate(intervals):
        # render remaining, or all if we're on the last scale level
        if idx != min_idx or scale == 0:
            # find position and scale
            min_interval, max_interval = intervals[idx]
            print(
                f"Fetching: {(data_scale, min_interval[0], min_interval[1], min_interval[2])}"
            )
            data = get_chunk(
                data_scale,
                int(min_interval[0] / 2**data_scale),
                int(min_interval[1] / 2**data_scale),
                int(min_interval[2] / 2**data_scale),
            ).transpose()
            node_scale = (
                2**data_scale,
                2**data_scale,
                2**data_scale,
            )
            viewer.add_image(
                data,
                scale=node_scale,
                translate=(min_interval[2], min_interval[1], min_interval[0]),
                name=f"chunk_{(data_scale, min_interval[0], min_interval[1], min_interval[2])}",
                blending="additive",
                opacity=0.8,
                rendering="mip",
            )
            # set data

    # recurse on closest
    if scale > 0:
        add_subnodes(intervals[min_idx], scale=scale - 1)

            
# use voxel resolution, might be 2x scale

# add_subnodes(((0, 0, 0), arrays[0].shape))
add_subnodes(view_interval, scale=1)

napari.run()
