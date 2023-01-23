from fibsem_tools.io import read_xarray

import napari

viewer = napari.Viewer()

def open_dataset(container: str, dataset: str):
    num_scales = 4
    # TODO num_scales needs to be read from the container/dataset
    arrays = [
        read_xarray(
            f"{container}/{dataset}/s{scale}/",
            storage_options={"anon": True},
        )
        for scale in range(num_scales)
    ]
    return arrays

container = 's3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5'
# When this container is displayed, the listing should match:
#   https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_mus-kidney/jrc_mus-kidney.n5/

def browse_container(container: str, viewer):
    """This function should display the contents of the container in a tree view widget.
    This widget might be useful:
      https://github.com/BodenmillerGroup/napari-hierarchical/blob/main/src/napari_hierarchical/widgets/_group_tree_view.py

    It should be possible to add context menu entries to the tree view that
    allows users to create functions of the form:
      my_function(container, dataset) which will be called on the corresponding tree view entry
    
    Arguments:
    - container: this is a string representing a path to a zarr
    """
    pass

browse_container(container, viewer)
# This should open the `browse_container` widget in napari
# Provide a simple right click context menu that uses `open_dataset` and prints the shapes of all arrays in a multiscale dataset
# for example, use:
# - container = "s3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5"
# - dataset = "labels/empanada-mito_seg"
# The shapes should be:
# - (11099, 3988, 6143)
# - (5549, 1994, 3071)
# - (2774, 997, 1535)
# - (1387, 498, 767)

