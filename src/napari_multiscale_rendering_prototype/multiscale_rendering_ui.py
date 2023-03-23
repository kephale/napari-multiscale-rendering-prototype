import os
from functools import partial
from pathlib import Path

import dask.array as da
import napari
import zarr
from fibsem_tools.io import read_xarray
from napari_hierarchical import controller
from napari_hierarchical.widgets._group_tree_model import QGroupTreeModel
from napari_hierarchical.widgets._group_tree_view import QGroupTreeView
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QMenu, QPushButton, QTextEdit, QVBoxLayout, QWidget

from .poor_octree import render_poor_octree


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

    return {
        "Print info": partial(print_image_info),
        "Show 2D": partial(show_image, multi_dim=False),
        "Show 3D": partial(show_image, multi_dim=True),
        "Poor Octree": partial(render_poor_octree),
    }


def _example_data():

    return {
        "Kidney example": "s3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5",  # noqa: E501
        "Luthi Zenodo example": "",
    }


class QtGroupTreeView(QGroupTreeView):
    def _on_custom_context_menu_requested(self, pos: QPoint) -> None:
        index = self.indexAt(pos)
        self.actions = []
        if index.isValid():
            group = index.internalPointer()
            menu = QMenu()
            readers = _get_available_renderers()
            render_fn_map = {}
            for reader, reader_fn in readers.items():
                action = menu.addAction(reader)
                self.actions.append(action)
                render_fn_map[action] = reader_fn
            result = menu.exec(self.mapToGlobal(pos))
            if result in self.actions:
                render_fn_map[result](group, controller.viewer)


def print_image_info(group, viewer):
    paths = []
    for array in group.iter_arrays(recursive=False):
        paths.append(array.zarr_path)

    if Path(array.zarr_file).suffix == ".n5":
        z = zarr.open(zarr.N5FSStore(array.zarr_file, anon=True))
    else:
        z = zarr.open(store=str(array.zarr_file), mode="r")

    for path in paths:
        zarr_array = z[path]
        print(zarr_array.shape)


def show_image(group, viewer, multi_dim=False):

    paths = []
    for array in group.iter_arrays(recursive=True):
        paths.append(array.zarr_path)

    if Path(array.zarr_file).suffix == ".n5":
        z = zarr.open(zarr.N5FSStore(array.zarr_file, anon=True))
    else:
        z = zarr.open(store=str(array.zarr_file), mode="r")

    data = []
    for path in paths:
        zarr_array = z[path]
        data.append(da.from_zarr(zarr_array, chunks=zarr_array.chunks))

    if multi_dim:
        viewer.dims.ndisplay = 3
        viewer.add_image(data, contrast_limits=(18000, 30000), multiscale=True)

    else:
        viewer.add_image(data, contrast_limits=(18000, 30000), multiscale=True)


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
        self._hide_delegate_icons()
        self.image_path = QTextEdit("Enter data location")
        self.example_button = QPushButton("Example Data")
        self.examples_menu = QMenu()
        self.example_button.setMenu(self.examples_menu)
        self.get_data_btn = QPushButton("Get Data")

        layout = QVBoxLayout()
        layout.addWidget(self.image_path)
        layout.addWidget(self.get_data_btn)
        layout.addWidget(self.example_button)
        layout.addWidget(self.treeWidget)
        self.setLayout(layout)

        self.image_path.setFixedHeight(40)
        self.get_data_btn.clicked.connect(self.get_data_from_url)

        def _get_example_data():
            action = self.sender()
            path = self.example_dict[action.text()]
            if action.text() == "Luthi Zenodo example":
                path = luethi_zenodo_7144919()
            self.image_path.setText(path)
            self.get_data_from_url()

        self.example_dict = _example_data()
        for ex, ex_path in self.example_dict.items():

            action = self.examples_menu.addAction(ex)
            action.triggered.connect(_get_example_data)

    def _hide_delegate_icons(self):

        itemDelegate = self.treeWidget.itemDelegateForColumn(
            QGroupTreeModel.COLUMNS.LOADED
        )
        itemDelegate._icon_size = (0, 0)

        self.treeWidget.setItemDelegateForColumn(
            QGroupTreeModel.COLUMNS.LOADED, itemDelegate
        )

        itemDelegate = self.treeWidget.itemDelegateForColumn(
            QGroupTreeModel.COLUMNS.VISIBLE
        )
        itemDelegate._icon_size = (0, 0)

        self.treeWidget.setItemDelegateForColumn(
            QGroupTreeModel.COLUMNS.VISIBLE, itemDelegate
        )

    def get_data_from_url(self, event=None):
        url = self.image_path.toPlainText()

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
    widget = MultiscaleWidget()
    widget.image_path.setText(container)
    widget.get_data_from_url()
    viewer.window.add_dock_widget(widget, name="Multiscale Widget")


def luethi_zenodo_7144919():
    import pooch

    # Downloaded from https://zenodo.org/record/7144919#.Y-OvqhPMI0R
    # TODO use pooch to fetch from zenodo
    dest_dir = pooch.retrieve(
        url="https://zenodo.org/record/7144919/files/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip?download=1",  # noqa: E501
        known_hash="e6773fc97dcf3689e2f42e6504e0d4f4d0845c329dfbdfe92f61c2f3f1a4d55d",  # noqa: E501
        processor=pooch.Unzip(),
    )
    local_container = os.path.split(dest_dir[0])[0]
    return local_container
