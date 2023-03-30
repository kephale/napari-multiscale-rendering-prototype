## Multiscale Widget Examples

This widget is a work in progress.  Readers may need to be written in order to view your data.  If you want to add a new renderer option to the drop down menu, see [here](#Add-renderer-to-widget-menu).

In order to run a couple of examples to view the multiscale widget, follow the steps listed here:

### Create a conda environment.

```
mamba env create -f environment.yml
mamba activate napari-multiscale
```

### Clone and install the `napari-multiscale-rendering-prototype` repo.

```
git clone git@github.com:ppwadhwa/napari-multiscale-rendering-prototype.git
cd napari-multiscale-rendering-prototype/src/napari_multiscale_rendering_prototype
pip install -e .
```

### Clone and install the `napari-hierarchical` repo from the following location and checkout the `update_zarr_reader` branch.

```
git clone git@github.com:ppwadhwa/napari-hierarchical.git
cd napari-hierarchical
git checkout update_zarr_reader

pip install -e .
```

### Run napari and use the multiscale widget:

   1. Start `napari`
   2. Select the `Plugins` menu
   3. Select `Multiscale Rendering (napari-multiscale-rendering-prototype)`
   4. Click on `Example Data` and you will see a few options.
   5. Right click on the image tree to see the renderer menu.

Note: Currently, for the `Poor Octree` renderer, it is working with the `Luthi Zenodo` example if you right click on the `/B/03/03` level.

In order to see 2D/3D, you will have to nagivate further into the tree.


##  Add renderer to widget menu.

In `multiscale_rendering_ui.py`, add an entry to the dictionary in the function `_get_available_renderers`.

   Use form:
   `"<Render Display Name>": partial(<renderer function>)`