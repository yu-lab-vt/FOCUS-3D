# FOCUS-3D

[![License BSD-3](https://img.shields.io/pypi/l/cellseg.svg?color=green)](https://github.com/Qinghua24/cellseg/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cellseg.svg?color=green)](https://pypi.org/project/cellseg)
[![Python Version](https://img.shields.io/pypi/pyversions/cellseg.svg?color=green)](https://python.org)
[![tests](https://github.com/Qinghua24/cellseg/workflows/tests/badge.svg)](https://github.com/Qinghua24/cellseg/actions)
[![codecov](https://codecov.io/gh/Qinghua24/cellseg/branch/main/graph/badge.svg)](https://codecov.io/gh/Qinghua24/cellseg)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/cellseg)](https://napari-hub.org/plugins/cellseg)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)

FOCUS-3D provides a user-friendly napari plugin for interactive 3D cell segmentation, manual curation, model fine-tuning, 3D reconstruction, and quantitative analysis. Users can run automatic 3D segmentation with pretrained FOCUS3D models, manually correct segmentation errors, perform one-click segmentation, prepare curated patches for human-in-the-loop fine-tuning, reconstruct selected 3D cell instances, and compute quantitative statistics within the same napari workflow.

<img width="3839" height="2082" alt="image" src="https://github.com/user-attachments/assets/2a9ccc08-3109-4b73-bcae-0514bcec2a86" />

## Installation

You can install `FOCUS-3D` with the following steps.

### 1. Create a new environment

```bash
cd your_path
conda env create -n focus3d -f environment.yml
```

### 2. Install FOCUS-3D

```bash
conda activate focus3d
pip install -U "focus-3d[all]"
pip install -e .
```

### 3. Launch napari

```bash
napari
```

## Usage

1. Open FOCUS-3D from `Plugins -> FOCUS-3D`.
2. Load the raw image from `File -> Open Folder` for a `.zarr` dataset, or `File -> Open File(s)` for `.tif` images.
3. Use the `Basic`, `Segmentation`, and `Analysis` tabs to load labels, run FOCUS3D segmentation, manually curate results, fine-tune models, reconstruct 3D structures, and calculate quantitative statistics.

## FOCUS-3D Overview

FOCUS-3D provides an integrated workflow for 3D microscopy image analysis. It combines automatic segmentation, interactive correction, local one-click refinement, curated patch generation, model fine-tuning, 3D label reconstruction, and quantitative measurement in a single napari-based interface. This design allows users to move smoothly from raw 3D images to corrected segmentation results and downstream analysis, while also supporting human-in-the-loop improvement of FOCUS3D models using newly curated annotations.

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

The `Segmentation` menu provides automatic 3D segmentation with **FOCUS3D**, one-click local segmentation refinement, and human-in-the-loop fine-tuning based on curated labels.
<img width="3839" height="2084" alt="image" src="https://github.com/user-attachments/assets/74b6b9ae-0a00-4b40-bb31-fb743709c725" />

### Run Segmentation

| Parameter / Operation | Description |
|---|---|
| Z Ratio | Set the physical Z-to-XY spacing ratio for anisotropic 3D data. |
| Output Path | Set the output folder for FOCUS3D segmentation results. By default, the plugin creates an output folder based on the input image path. |
| GPU IDs | Select the GPU device used for inference, such as `0`, `1`, or `0,1`. |
| Refresh | Refresh the list of available GPU devices. |
| Advanced | Expand or collapse advanced inference settings. |
| Checkpoint | Select the pretrained FOCUS3D model checkpoint. The default path is relative to the FOCUS3D backend folder. |
| Configure | Select the FOCUS3D configuration file. The default configuration is `configs/3d_test.yaml`. |
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
| Run 3D Segmentation | Run automatic 3D segmentation using FOCUS3D. The result is loaded back into napari as an editable label layer, and the confidence map can also be saved or visualized. |

### One-click Segmentation

| Operation | Description |
|---|---|
| Enter Inactive Mode | Load the one-click segmentation model and enter interactive local segmentation mode. After entering this mode, users can click on a target cell region to trigger local refinement. |
| Exit Inactive Mode | Exit one-click segmentation mode and return to normal interaction. |
| Status Label | Display the current status of the one-click segmentation module, such as inactive, loading, active, or busy. |
| Local Refinement | Use the pretrained FOCUS3D model to refine a clicked cell or local region, supporting fast correction of missed or inaccurate instances. |
| Undo | Undo the latest one-click segmentation refinement result with `Ctrl + Z`. |

### Finetune with Current Labels

| Operation | Description |
|---|---|
| Calculate Valid Patches | Automatically scan the current image and label volume to find valid 3D patches for fine-tuning. Patches are selected according to patch size, stride, and intensity threshold. |
| Select Patch ID | Select a valid patch by ID. The corresponding patch region is highlighted in the viewer. |
| Save Path | Set the folder for saving curated training patches. The expected structure is `imagesTr/` and `labelsTr/`. |
| Curate Selected Patch | Open the selected 3D patch in a new napari viewer for focused manual correction. |
| Save Curated Patch | Save the curated patch image and label into `imagesTr/` and `labelsTr/`. |
| Checkpoint Dir | Set the output directory for fine-tuned FOCUS3D checkpoints. |
| Run Fine-tune | Fine-tune the FOCUS3D model using the curated patches. After fine-tuning, the newly generated checkpoint can be used for subsequent segmentation. |

## Analysis Menu

The `Analysis` menu provides tools for 3D structure reconstruction, full-volume 3D visualization, and quantitative statistics.

<img width="3839" height="2087" alt="image" src="https://github.com/user-attachments/assets/7d499c7c-952c-4ad0-a2a4-5f1b9896d0f3" />

### 3D Label Reconstruction

| Operation | Description |
|---|---|
| Z Ratio | Set the physical Z-to-XY ratio for 3D reconstruction. This is important for anisotropic microscopy volumes. |
| Reconstruct Selected Label | Reconstruct the selected label as a 3D mesh for visualization and downstream analysis. |
| Load Mesh | Load a previously saved 3D mesh file. |
| Save Mesh | Save the reconstructed 3D mesh. |

### Full 3D View

| Operation | Description |
|---|---|
| Z Ratio | Set the Z-axis scaling factor for full-volume 3D visualization. |
| Switch to 3D View | Switch the current napari viewer between 2D slice view and 3D visualization mode. |

### Quantitative Statistics

| Operation | Description |
|---|---|
| Calculate Size Distribution | Calculate cell count and cell size distribution from the current label layer. This can be used for basic quantitative analysis of segmentation results. |

## Recommended Workflow

A typical workflow is:

1. Load the raw 3D microscopy image into napari.
2. Run automatic 3D segmentation with **FOCUS3D** from the `Segmentation` menu.
3. Inspect the segmentation result in the `Basic` menu.
4. Correct segmentation errors using manual curation tools such as **Add to Label**, **Subtract from Label**, **Add New Label**, and **Delete Inside ROI**.
5. Use **One-click Segmentation** for fast local correction of difficult or missed cells.
6. Save the curated labels as `.zarr` or `.tif`.
7. Optionally select valid curated patches and fine-tune FOCUS3D with the corrected labels.
8. Use the `Analysis` menu for 3D reconstruction, full 3D visualization, and quantitative statistics.

## License

Distributed under the terms of the [BSD-3] license, `FOCUS-3D` is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description or contact zhangqh24@mails.tsinghua.edu.cn.

[napari]: https://github.com/napari/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
