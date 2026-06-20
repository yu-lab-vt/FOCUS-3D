# FOCUS-3D

[![License BSD-3](https://img.shields.io/pypi/l/cellseg.svg?color=green)](https://github.com/Qinghua24/cellseg/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cellseg.svg?color=green)](https://pypi.org/project/cellseg)
[![Python Version](https://img.shields.io/pypi/pyversions/cellseg.svg?color=green)](https://python.org)
[![tests](https://github.com/Qinghua24/cellseg/workflows/tests/badge.svg)](https://github.com/Qinghua24/cellseg/actions)
[![codecov](https://codecov.io/gh/Qinghua24/cellseg/branch/main/graph/badge.svg)](https://codecov.io/gh/Qinghua24/cellseg)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/cellseg)](https://napari-hub.org/plugins/cellseg)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)

FOCUS-3D provides a user-friendly napari plugin for interactive 3D cell segmentation, manual curation, model fine-tuning, and analysis. Users can run automatic 3D segmentation with pretrained FOCUS-3D models, manually correct segmentation errors, perform one-click segmentation, prepare curated patches for human-in-the-loop fine-tuning, reconstruct selected 3D cell instances, and compute quantitative statistics within the same napari workflow. Our website is [https://www.quiclab.org.cn/focus-3d](https://www.quiclab.org.cn/focus-3d).

<img width="800" height="434" alt="image" src="https://github.com/user-attachments/assets/2a9ccc08-3109-4b73-bcae-0514bcec2a86" />

## Installation

### Windows

#### 1. Create a new environment

```bash
conda create -n focus3d python=3.10 -y
conda activate focus3d
```

#### 2. Install FOCUS-3D

```bash
pip install -U "focus-3d[all]"
```

#### 3. Launch napari

```bash
python -m napari
```

---

### Linux

#### 1. Create a new environment

```bash
conda create -n focus3d python=3.10 -y
conda activate focus3d
```

#### 2. Install PyTorch

For CUDA 12.6:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

If your CUDA version is different, please install the corresponding PyTorch version from the official PyTorch installation guide.

#### 3. Install FOCUS-3D

```bash
pip install -U "focus-3d[all]"
```

#### 4. Install detectron2

```bash
pip install "git+https://github.com/facebookresearch/detectron2.git"
```

#### 5. Launch napari

```bash
python -m napari
```

## Usage

1. Open FOCUS-3D from `Plugins -> FOCUS-3D`.
2. Load the raw image from `File -> Open Folder` for a `.zarr` dataset(recommend), or `File -> Open File(s)` for `.tif` images.
3. Use the `Basic`, `Segmentation`, and `Analysis` tabs to load labels, run segmentation, manually curate results, fine-tune models, reconstruct 3D structures, and calculate quantitative statistics.

## Basic Menu

The `Basic` menu provides functions for loading segmentation labels, adjusting image display, manually curating labels, and saving edited results.

### Load Label

| Operation | Description |
|---|---|
| Load from Zarr | Load an existing 3D label map from a `.zarr` folder. This is recommended for large 3D volumes because it supports efficient data access and editing. |
| Load from TIFF | Load an existing 3D label map from a `.tif` or `.tiff` file. |
| Only Show Contour | Display label contours instead of filled label masks, making it easier to inspect boundaries and compare labels with the raw image. |

### Display Settings

| Operation | Description |
|---|---|
| Minimum / Maximum | Adjust the display contrast range of the currently selected image layer. |
| Auto | Automatically estimate a suitable contrast range for the active image layer using robust intensity percentiles. |
| Reset Contrast | Reset the contrast range of the active image layer. |
| Channel Composite | Select one or more image layers and assign different colormaps for multi-channel visualization. |
| Apply | Apply the selected channel visibility and colormap settings. |
| Reset Display | Reset image layers to the default display mode. |

### Manual Curation

| Operation | Shortcut | Description |
|---|---|---|
| Enter Curation Mode | — | Activate label editing mode. Users can click on an existing label to select and modify it. |
| Polygon Draw Mode | — | Define an ROI by clicking polygon vertices. The enclosed region is used for label editing. |
| Brush Draw Mode | — | Define an ROI by painting directly on the image. The painted region is used for label editing. |
| Trace Draw Mode | — | Define an ROI using trace-based drawing. |
| Brush Size | `+` / `-` | Adjust the ROI brush size or label brush size. |
| Brush Color | — | Change the display color of the ROI brush. |
| Add to Label | `A` | Add the selected ROI to the currently selected label. |
| Subtract from Label | `S` | Remove the selected ROI from the currently selected label. |
| Add New Label | `Ctrl + D` | Create a new label. Regions from multiple Z slices can be combined into the same 3D instance. |
| Change Label | `C` | Change the ID of the currently selected label. |
| Apply ROI | `Ctrl + A` | Apply the current ROI operation. Used with **Add to Label**, **Subtract from Label**, and **Add New Label**. |
| Cancel ROI | `Ctrl + C` | Cancel the current ROI operation without applying changes. |
| Delete Current Z | `Delete` | Delete the selected label only on the current Z slice. |
| Delete All Z | `Ctrl + Delete` | Delete the selected label across all Z slices. |
| Delete Inside ROI (All Z) | — | Delete all labels inside the selected ROI across all Z slices. |
| Show Log File | — | Open or show the curation log file that records editing operations. |
| Previous Slice | `Q` | Move to the previous Z slice. |
| Next Slice | `W` | Move to the next Z slice. |
| Undo Local Refinement | `Ctrl + Z` | Undo the latest one-click local refinement result when available. |

### Save

| Operation | Description |
|---|---|
| Browse | Select the folder for saving the current label result. |
| Save format | Choose the output format, including `Zarr (recommended)` or `TIFF`. |
| Update Labels | Reassign label IDs before saving so that labels are stored in a clean consecutive order. |
| Save | Save the current label layer to the selected output path. |

## Segmentation Menu

The `Segmentation` menu provides automatic 3D segmentation with **FOCUS-3D**, one-click local segmentation refinement, and human-in-the-loop fine-tuning based on curated labels.
<img width="800" height="434" alt="image" src="https://github.com/user-attachments/assets/74b6b9ae-0a00-4b40-bb31-fb743709c725" />

### Run Segmentation

| Parameter / Operation | Description |
|---|---|
| Z Ratio | Set the physical Z-to-XY spacing ratio for anisotropic 3D data. |
| Output Path | Set the output folder for FOCUS-3D segmentation results. By default, the plugin creates an output folder based on the input image path. |
| GPU IDs | Select the GPU device used for inference, such as `0`, `1`, or `0,1`. |
| Refresh | Refresh the list of available GPU devices. |
| Advanced | Expand or collapse advanced inference settings. |
| Checkpoint | Select the pretrained model checkpoint. The default path is relative to the backend folder. |
| Configure | Select the configuration file. The default configuration is `configs/3d_test.yaml`. |
| Lower / Upper Percentile | Set the intensity percentile range for image normalization before inference. |
| Patch size (Z/Y/X) | Set the 3D patch size used for sliding-window inference. |
| Stride (Z/Y/X) | Set the stride between neighboring inference patches. Smaller strides increase overlap but require more computation. |
| Background intensity | Ignore patches whose maximum intensity is below this threshold. |
| Batch size | Set the inference batch size. Larger values may improve speed but require more GPU memory. |
| Score confidence | Set the object-level confidence threshold for accepting predicted instances. |
| Mask confidence | Set the voxel-level mask confidence threshold for generating instance masks. |
| Min area (2D) | Remove small 2D components below the specified area threshold during stitching or post-processing. |
| Min size (3D) | Remove 3D objects smaller than the specified voxel size. A value of `0` disables minimum-size filtering. |
| Max size (3D) | Remove 3D objects larger than the specified voxel size. The default value is `100000`. |
| Run 3D Segmentation | Run automatic 3D segmentation using FOCUS-3D. The result is loaded back into napari as an editable label layer, and the confidence map can also be saved or visualized. |

### One-click Segmentation

| Operation | Description |
|---|---|
| Enter Inactive Mode | Load the one-click segmentation model and enter interactive local segmentation mode. After entering this mode, users can click on a target cell region to trigger local refinement. |
| Exit Inactive Mode | Exit one-click segmentation mode and return to normal interaction. |
| Status Label | Display the current status of the one-click segmentation module, such as inactive, loading, active, or busy. |
| Local Refinement | Use the pretrained FOCUS-3D model to refine a clicked cell or local region, supporting fast correction of missed or inaccurate instances. |
| Undo | Undo the latest one-click segmentation refinement result with `Ctrl + Z`. |

### Finetune with Current Labels

| Operation | Description |
|---|---|
| Calculate Valid Patches | Automatically scan the current image and label volume to find valid 3D patches for fine-tuning. Patches are selected according to patch size, stride, and intensity threshold. |
| Select Patch ID | Select a valid patch by ID. The corresponding patch region is highlighted in the viewer. |
| Save Path | Set the folder for saving curated training patches. The expected structure is `imagesTr/` and `labelsTr/`. |
| Curate Selected Patch | Open the selected 3D patch in a new napari viewer for focused manual correction. |
| Save Curated Patch | Save the curated patch image and label into `imagesTr/` and `labelsTr/`. |
| Checkpoint Dir | Set the output directory for fine-tuned FOCUS-3D checkpoints. |
| Run Fine-tune | Fine-tune the FOCUS-3D model using the curated patches. After fine-tuning, the newly generated checkpoint can be used for subsequent segmentation. |

## Analysis Menu

The `Analysis` menu provides tools for selected-label 3D reconstruction, full-volume 3D visualization, and task-based morphometry analysis.

<img width="800" height="434" alt="image" src="https://github.com/user-attachments/assets/bee8aa3e-7f57-439d-93bc-cf28ebbb0f9c" />


### 3D Label Reconstruction

This module reconstructs one selected non-background label into a 3D surface mesh. The mesh is generated from the selected label mask and displayed in a separate napari 3D viewer as a `Surface` layer. The Z scaling factor is applied during mesh generation, which is important for anisotropic microscopy volumes.

| Operation                  | Description                                                                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Z Ratio                    | Set the physical Z-to-XY scaling ratio used during mesh reconstruction. A larger value stretches the reconstructed mesh along the Z axis.                    |
| Reconstruct Selected Label | Reconstruct the currently selected non-background label into a 3D surface mesh. The selected label is cropped before mesh extraction to reduce memory usage. |
| Load Mesh                  | Load a previously saved `.npz` mesh file and display it in a separate 3D napari viewer.                                                                      |
| Save Mesh                  | Save the latest reconstructed mesh as a compressed `.npz` file, including vertices, faces, voxel count, Z ratio, and label ID.                               |

### Full 3D View

This module switches the main napari viewer between 2D slice mode and full-volume 3D visualization mode. It applies anisotropic Z scaling to spatial layers before entering 3D view.

| Operation                             | Description                                                                                                                                                           |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Z Ratio                               | Set the Z-axis scaling factor for full-volume 3D visualization. For 3D data ordered as `(Z, Y, X)`, the layer scale is set to `(Z Ratio, 1.0, 1.0)`.                  |
| Switch to 3D View / Switch to 2D View | Toggle the main viewer between 2D slice view and 3D visualization. When entering 3D mode, the plugin also switches layers to pan/zoom mode to enable camera rotation. |

### Quick Quantitative Statistics

This module provides a lightweight size-distribution summary from the current label layer. It is intended for a quick overview of segmentation results.

| Operation                   | Description                                                                                                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Calculate Size Distribution | Count all non-background labels and compute their voxel sizes. The result dialog reports the number of cells, minimum size, maximum size, and a histogram of cell size distribution. |

### Morphometry Analysis

The `Morphometry Analysis` panel provides task-based quantitative analysis for 3D label layers. Each task runs independently in the background and saves its results to a task-specific output folder.

> Morphometry analysis currently supports 3D label layers. Raw image layers are only required when intensity-based features are selected.

#### Common Settings

| Operation              | Description                                                                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Voxel size (Z / Y / X) | Set the physical voxel size used for volume, surface area, centroid, distance, and shape measurements.                                                   |
| Output folder          | Set the root folder for morphometry outputs. Each task writes results into a subfolder such as `basic_info`, `neighborhood`, `contact`, or `clustering`. |
| Browse                 | Select an output folder from the file system.                                                                                                            |
| Open Output Folder     | Open the current morphometry output folder.                                                                                                              |

#### 1. Basic Information

Compute selected per-cell measurements and save them into one CSV table.

| Operation             | Description                                                                                                                                                                                         |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Raw layer             | Select the raw image layer used for intensity measurements. This is required only when intensity features are selected.                                                                             |
| Refresh Raw Layers    | Refresh the raw-layer dropdown from the current napari viewer.                                                                                                                                      |
| Features to compute   | Select cell-level features to calculate, including volume, equivalent diameter, surface area, centroid, sphericity, compactness, major axis length, elongation, flatness, and intensity statistics. |
| Run Basic Information | Compute the selected features and save `basic_info_cell_features.csv` and `basic_info_summary.csv`.                                                                                                 |
| Show feature          | Display a selected computed feature as a mapped 3D image layer in napari. Non-map features such as centroid and some scalar-only fields are saved in CSV but are not shown as feature maps.         |

Supported basic features include:

| Feature                          | Description                                                                         |
| -------------------------------- | ----------------------------------------------------------------------------------- |
| Volume                           | Physical cell volume computed from voxel count and voxel size.                      |
| Equivalent diameter              | Diameter of a sphere with the same volume.                                          |
| Surface area                     | Surface area estimated from exposed voxel faces.                                    |
| Centroid                         | Cell centroid in physical coordinates.                                              |
| Sphericity                       | Shape compactness relative to a sphere.                                             |
| Compactness                      | Surface-area-to-volume based compactness measurement.                               |
| Axis major                       | Major axis length estimated from PCA on physical voxel coordinates.                 |
| Elongation                       | Ratio describing long-axis elongation.                                              |
| Flatness                         | Ratio describing flattening along the minor axis.                                   |
| Min / Max / Mean / Std intensity | Intensity statistics computed inside each labeled cell. Requires a raw image layer. |

#### 2. Neighborhood Analysis

Compute centroid-based neighborhood features using either k-nearest neighbors or a radius-based neighborhood.

| Operation            | Description                                                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Mode                 | Choose between `kNN distance` and `Radius count / density`.                                                                          |
| k                    | Number of neighbors used in kNN mode.                                                                                                |
| Radius pixels        | Radius used in radius mode, measured in voxel/pixel coordinates.                                                                     |
| Run Neighborhood     | Compute neighborhood features and save `neighborhood_cell_features.csv` and `neighborhood_summary.csv`.                              |
| Show feature         | Display a selected neighborhood feature as a mapped 3D image layer.                                                                  |
| Run Local Comparison | After neighborhood analysis is finished, compute a local z-score for one selected feature by comparing each cell with its neighbors. |

In `kNN distance` mode, the plugin computes nearest-neighbor distance, mean/median kNN distance, and local density. In `Radius count / density` mode, it computes the number of neighboring cells within the selected radius and the corresponding local density.

Local comparison supports features such as mean intensity, sphericity, volume, surface area, compactness, elongation, and flatness. The output includes the original feature value, local neighbor mean, local neighbor standard deviation, and local z-score.

#### 3. Contact Graph Analysis

Compute face-touching contact relationships between neighboring labeled cells.

| Operation           | Description                                                                                                                                        |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Features to compute | Select contact features to calculate, including neighbor count, total contact area, mean contact area, maximum contact area, and contact fraction. |
| Run Contact Graph   | Compute the contact graph and save `contact_cell_features.csv`, `contact_edges.csv`, and `contact_summary.csv`.                                    |
| Show feature        | Display a selected contact feature as a mapped 3D image layer.                                                                                     |

Contact area is estimated from shared voxel faces using the physical voxel size. `contact_edges.csv` stores pairwise cell-cell contact edges, while `contact_cell_features.csv` stores per-cell contact summaries.

#### 4. Clustering

Cluster cells using one selected feature.

| Operation      | Description                                                                                                                                                                     |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Feature        | Select the feature used for clustering. Available features include morphology, intensity, and contact-related measurements.                                                     |
| Clusters       | Set the number of clusters for K-means clustering.                                                                                                                              |
| Run Clustering | Cluster cells based on the selected feature and save `clustering_cell_features.csv` and `clustering_summary.csv`. The `cluster_id` result is displayed automatically in napari. |

If an intensity feature is selected for clustering, a raw image layer is required. Clustering uses the selected feature values and assigns each valid cell a `cluster_id`.

## Recommended Workflow

A typical workflow is:

1. Load the raw 3D microscopy image into napari.
2. Run automatic 3D segmentation with **FOCUS-3D** from the `Segmentation` menu.
3. Inspect the segmentation result in the `Basic` menu.
4. Correct segmentation errors using manual curation tools such as **Add to Label**, **Subtract from Label**, **Add New Label**, and **Delete Inside ROI**.
5. Use **One-click Segmentation** for fast local correction of difficult or missed cells.
6. Save the curated labels as `.zarr` or `.tif`.
7. Optionally select valid curated patches and fine-tune FOCUS-3D with the corrected labels.
8. Use the `Analysis` menu for 3D reconstruction, full 3D visualization, and quantitative statistics.

## License

Distributed under the terms of the [BSD-3] license, `FOCUS-3D` is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description or contact zhangqh24@mails.tsinghua.edu.cn.

[napari]: https://github.com/napari/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

## Citing
```bibtex
@article{wang2026high,
  title={High-Fidelity Long-term Whole-embryo Lineage and Fate Reconstruction by Iterative Tracking with Error Correction},
  author={Wang, Mengfan and Zhang, Qinghua and Wang, Congchao and Chi, Yunfeng and Zheng, Wei and Mu, Zeyu and Cao, Xiangyu and Zhang, Weizhan and Yang, Boao and Schier, Alexander F. and Acedo, Joaquin Navajas and Wan, Yinan and Yu, Guoqiang},
  year={2026},
  doi={10.64898/2026.03.12.711203}
}
```
