import dask.array as da
import napari
import zarr
from fibsem_tools.io import read_xarray
from napari_hierarchical import controller
from napari_hierarchical.widgets._group_tree_view import QGroupTreeView
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QMenu, QPushButton, QTextEdit, QVBoxLayout, QWidget


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


container = "s3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5"
# When this container is displayed, the listing should match:
#   https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_mus-kidney/jrc_mus-kidney.n5/


def _get_available_renderers():

    return ["Print info", "Show 2D", "Show 3D"]


class QtGroupTreeView(QGroupTreeView):
    def _on_custom_context_menu_requested(self, pos: QPoint) -> None:
        index = self.indexAt(pos)
        self.actions = []
        if index.isValid():
            group = index.internalPointer()
            # assert isinstance(group, Group)
            menu = QMenu()
            readers = _get_available_renderers()
            for reader in readers:
                self.actions.append(menu.addAction(reader))
            result = menu.exec(self.mapToGlobal(pos))
            if result == self.actions[0]:
                print_image_info(group)

            if result == self.actions[1]:
                show_2d(group, controller.viewer)
            if result == self.actions[2]:
                show_3d(group, controller.viewer)


def print_image_info(group):
    # container == array.zarr_file
    # path = dataset
    paths = []
    for array in group.iter_arrays(recursive=True):

        paths.append(array.zarr_path)
    z = zarr.open(zarr.N5FSStore(array.zarr_file, anon=True))
    for path in paths:
        zarr_array = z[path]
        print(zarr_array.shape)


def show_2d(group, viewer):
    cnt = 0
    data = []
    for array in group.iter_arrays(recursive=True):
        path = array.zarr_path
        if cnt == 0:
            z = zarr.open(zarr.N5FSStore(array.zarr_file, anon=True))
        zarr_array = z[path]
        data.append(da.from_zarr(zarr_array, chunks=zarr_array.chunks))

    viewer.add_image(data, contrast_limits=(18000, 3000), multiscale=True)


def show_3d(group, viewer):
    cnt = 0
    data = []
    for array in group.iter_arrays(recursive=True):
        path = array.zarr_path
        if cnt == 0:
            z = zarr.open(zarr.N5FSStore(array.zarr_file, anon=True))

        zarr_array = z[path]
        data.append(da.from_zarr(zarr_array, chunks=zarr_array.chunks))

    viewer.dims.ndisplay = 3

    viewer.add_image(data, contrast_limits=(18000, 3000), multiscale=True)


class MultiscaleWidget(QWidget):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        viewer = controller.viewer or napari.current_viewer()
        assert viewer is not None
        if controller.viewer != viewer:
            controller.register_viewer(viewer)

        self.treeWidget = QtGroupTreeView(controller)
        self.hrl_path = QTextEdit("Enter data location")
        self.get_data_btn = QPushButton("Get Data")

        layout = QVBoxLayout()
        layout.addWidget(self.hrl_path)
        layout.addWidget(self.get_data_btn)
        layout.addWidget(self.treeWidget)
        self.setLayout(layout)

        self.hrl_path.setFixedHeight(40)
        self.get_data_btn.clicked.connect(self.get_data_from_url)

    def get_data_from_url(self, event):
        url = self.hrl_path.toPlainText()

        controller.read_group(url)


def browse_container(container: str, viewer):
    """This function should display the contents of the container in a
    tree view widget.
    This widget might be useful:
      https://github.com/BodenmillerGroup/napari-hierarchical/blob/main/src/napari_hierarchical/widgets/_group_tree_view.py

    It should be possible to add context menu entries to the tree view that
    allows users to create functions of the form:
      my_function(container, dataset) which will be called on the corresponding
      tree view entry

    Arguments:
    - container: this is a string representing a path to a zarr
    """


# browse_container(container, viewer)
# This should open the `browse_container` widget in napari
# Provide a simple right click context menu that uses `open_dataset` and
# prints the shapes of all arrays in a multiscale dataset
# for example, use:
# - container = "s3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5"
# - dataset = "labels/empanada-mito_seg"
# The shapes should be:
# - (11099, 3988, 6143)
# - (5549, 1994, 3071)
# - (2774, 997, 1535)
# - (1387, 498, 767)
