# FOCUS-3D Interface Reference

This document describes the controls currently exposed by the FOCUS-3D napari plugin.

For a beginner-oriented walkthrough, return to the [Recommended Workflow](../README.md#recommended-workflow).

## Interface Overview

Open the plugin from:

```text
Plugins -> 3D Segmentation (FOCUS-3D)
```

The dock widget contains three tabs:

| Tab | Primary purpose |
|---|---|
| `Basic` | Load or create labels, adjust image display, manually curate labels, and save corrected results. |
| `Segmentation` | Run FOCUS-3D inference, perform one-click local refinement, and export curated patches for fine-tuning. |
| `Analysis` | Reconstruct selected cells, inspect full volumes in 3D, and calculate morphometric or spatial features. |

## Basic Tab

The `Basic` tab is the main workspace for inspecting and editing label volumes.

### Load Label

| Control | Description |
|---|---|
| `Load from Zarr` | Load an existing 3D label map from a `.zarr` directory. This format is recommended for large volumes and repeated editing. |
| `Load from TIFF` | Load a 3D label map from a `.tif` or `.tiff` file. |
| `Create Empty Label` | Create an all-zero editable label layer with the same shape as the active 3D image. The label data are stored as `curation_labels.zarr` under the current segmentation output directory. |
| `Only Show Contour` | Switch label layers between filled masks and contour-only display. This changes visualization but not label data. |

`Create Empty Label` requires a three-dimensional image layer with shape `(Z, Y, X)`.

### Display Settings

Display controls operate on the active image layer.

| Control | Description |
|---|---|
| `Minimum` | Set the lower display limit of the active image layer. |
| `Maximum` | Set the upper display limit of the active image layer. |
| `Auto` | Estimate display limits from robust intensity percentiles of a sampled subset of the active image. |
| `Reset Contrast` | Reset the active image to its sampled minimum and maximum intensity range. |
| `Channel Composite` | List the image layers currently loaded in napari. Each row provides a visibility checkbox and colormap selector. |
| `Apply` | Show the selected image layers using their assigned colormaps and additive blending. |
| `Reset Display` | Make all image layers visible and restore gray, translucent display. |

These settings do not modify the underlying image intensities.

### Manual Curation

Click `Enter Curation Mode` before selecting or editing an existing instance.

#### Draw modes

| Control | Description |
|---|---|
| `Polygon` | Draw an ROI by placing polygon vertices. |
| `Brush` | Paint an ROI directly on the current Z slice. |
| `Trace` | Draw a trace-based ROI. |
| `Brush Size` | Set the ROI or label brush diameter. The default ROI brush size is `2`. |
| `Brush Color` | Change the display color of the temporary brush ROI. |

#### Label-editing operations

| Control | Shortcut | Description |
|---|---:|---|
| `Enter Curation Mode` / `Exit Curation Mode` | — | Toggle label selection and editing. In curation mode, click an instance to select it. |
| `Add to Label` | `A` | Add the subsequently drawn ROI to the selected label. |
| `Subtract from Label` | `S` | Remove the subsequently drawn ROI from the selected label. |
| `Add New Label` | `Ctrl + D` | Allocate a new label ID. ROIs drawn on multiple Z slices can be added to the same new 3D instance before the operation is completed. |
| `Change Label` | `C` | Change the ID of the selected label. |
| `Apply ROI` | `Ctrl + A` | Apply the active add, subtract, or new-label ROI operation. |
| `Cancel ROI` | `Ctrl + C` | Discard the active ROI operation. |
| `Delete Current Z` | `Delete` | Delete the selected label only on the current Z slice. |
| `Delete All Z` | `Ctrl + Delete` | Delete the selected label throughout the 3D volume. |
| `Delete Inside ROI (All Z)` | — | Delete all label voxels inside the selected ROI across Z slices. |
| `Show Log File` | — | Show the location of the session curation log. |

During ROI editing, keep the intended segmentation label layer active. Temporary ROI and brush layers are created and removed automatically.

### Save

| Control | Description |
|---|---|
| Output path field | Directory in which the current label result will be saved. |
| `Browse` | Select the output directory. |
| `Save format` | Choose `Zarr (recommended)` or `TIFF`. |
| `Update Labels` | Renumber non-background instances into a clean consecutive sequence before saving. Enabled by default. |
| `Save` | Save the active/current label result. |

Use Zarr for large editable volumes. TIFF is convenient for exchange with software that does not read Zarr.

## Segmentation Tab

The `Segmentation` tab contains `Run Segmentation`, `One-click segmentation`, and `Finetune with Current Labels`.

<img width="800" height="434" alt="FOCUS-3D segmentation tab" src="https://github.com/user-attachments/assets/74b6b9ae-0a00-4b40-bb31-fb743709c725" />

## Run Segmentation

FOCUS-3D segments the active three-dimensional image layer. Select the intended image layer in napari before clicking `Run 3D Segmentation`.

### Frequently used settings

These controls remain visible outside the collapsed `Advanced` section.

| Parameter | Default | Description |
|---|---:|---|
| `Z Ratio` | `1.0` | Physical Z-to-XY spacing ratio. Use `1.0` for isotropic data. |
| `Output Path` | Automatic | Destination for the current inference run. The plugin generates a timestamped folder based on the active image path or layer name when possible. |
| `Checkpoint` | Project default | FOCUS-3D checkpoint used for inference. Relative paths are resolved from the FOCUS-3D backend directory. |
| `Cell radius (pixel)` | `15.0` | Estimated cell radius in the XY plane. The inference input is resampled so that the effective cell radius is approximately 15 pixels. |
| `Background intensity` | `1.0` | Skip a patch when its maximum raw intensity is below this threshold. |
| `Min size (3D)` | `0` | Remove instances smaller than this number of voxels. `0` disables minimum-size filtering. |
| `Max size (3D)` | `100000` | Remove instances larger than this number of voxels. |
| `Run 3D Segmentation` | — | Start inference in a background worker and display progress. The result is loaded as an editable label layer. |

`Max size (3D)` must be greater than or equal to `Min size (3D)`.

### Advanced

Click the arrow next to `Advanced` to expose model, normalization, patch-processing, and stitching controls.

#### Model and Config

| Parameter | Default | Description |
|---|---:|---|
| `Configure` | `configs/3d_test.yaml` | Configuration file used by the inference backend. Relative paths are resolved from the FOCUS-3D backend directory. |
| `GPU IDs` | First detected GPU | PyTorch-visible CUDA device index, such as `0`, `1`, or `0,1`. Current in-process inference uses the first selected ID unless the backend explicitly implements multi-GPU inference. |
| `Refresh` | — | Re-detect CUDA devices and rebuild the GPU list. |

When `CUDA_VISIBLE_DEVICES` was set before napari was launched, `cuda:0` refers to the first GPU visible to the current process and may not match physical GPU 0 in `nvidia-smi`.

#### Normalization

| Parameter | Default | Description |
|---|---:|---|
| `Lower` | `1.0` | Lower percentile for intensity normalization. |
| `Upper` | `99.0` | Upper percentile for intensity normalization. |

The lower percentile must remain below the upper percentile.

#### Patch processing

| Parameter | Default | Description |
|---|---:|---|
| `Stride (Z/Y/X)` | `24 / 64 / 64` | Step between neighboring inference patches. Smaller strides increase overlap and usually increase runtime and memory traffic. |
| `Batch size` | `16` | Number of patches processed together. Increase only when sufficient GPU memory is available. |

The inference patch size is fixed by the current model/configuration; only the stride is exposed in the interface.

#### Stitch

| Parameter | Default | Description |
|---|---:|---|
| `Score confidence` | `0.7` | Object-level confidence threshold used to retain predicted instances. |
| `Mask confidence` | `0.5` | Voxel-level mask threshold used to generate binary instance masks. |
| `Min area (2D)` | `64` | Remove small two-dimensional components during stitching or post-processing. |

### Segmentation outputs

After a successful run, the plugin:

1. loads the instance result as an editable napari `Labels` layer;
2. creates `curation_labels.zarr` in the inference output directory;
3. stores the original instance map, confidence map, and inference log when returned by the backend;
4. records metadata in the label layer so later curation and patch export can find the editable Zarr data.

## One-click Segmentation

One-click segmentation performs local model-based correction around a clicked cell or error region.

| Control | Description |
|---|---|
| `Enter Inactive Mode` | Current UI label for starting the module. It loads the one-click model and installs the viewer click callback. |
| `Exit Inactive Mode` | Remove the interactive callback and leave local-refinement mode. |
| Status text | Reports states such as inactive, loading, active, or busy. |
| Click in viewer | Trigger local refinement around the clicked target. |
| Undo | Press `Ctrl + Z` to undo the latest accepted local refinement. |

Keep a raw image layer and the main segmentation label layer loaded. The selected GPU and checkpoint settings are reused by the local inference path where applicable.

## Finetune with Current Labels

This panel prepares curated training samples. Model training itself is run from `notebooks/02_finetune.ipynb`, not from a GUI `Run Fine-tune` button.

| Control | Description |
|---|---|
| `Calculate Valid Patches` | Scan the current image and label volume for valid `(32, 96, 96)` patches. The scan uses the current segmentation stride and `Background intensity` threshold. |
| `Clear Patch Boxes` | Remove all patch overlays, clear the patch-selection state, and reactivate the main segmentation label layer. |
| `Select Patch ID` | Choose a valid patch numerically. The selected patch is emphasized with a thicker white boundary. |
| Patch-box click | Click inside a displayed patch rectangle to update `Select Patch ID`. |
| `Save Path` | Root directory for exported training data. The default is `<segmentation_output>/curated_patch`. |
| `Browse` | Select another patch-export directory. |
| `Curate Selected Patch` | Open the selected image and label crop in a new napari viewer. |
| Collapsed `Fine-tune` row | Show the instruction to run `notebooks/02_finetune.ipynb` after curated patches have been saved. |

### Patch curation viewer

The separate patch viewer contains:

- the cropped raw image;
- the cropped label map;
- the `Basic` tab curation controls;
- a `Save Curated Patch` panel.

The regular `Save` panel is hidden in this viewer. Use the `Save` button inside `Save Curated Patch` to export the pair.

Files are written as:

```text
<save_path>/
├── imagesTr/
│   └── patch_<ID>.tif
└── labelsTr/
    └── patch_<ID>.tif
```

The current implementation formats IDs with four digits, for example `patch_0001.tif`.

## Analysis Tab

The `Analysis` tab provides selected-cell reconstruction, full-volume visualization, and task-based morphometry.

<img width="800" height="434" alt="FOCUS-3D analysis tab" src="https://github.com/user-attachments/assets/bee8aa3e-7f57-439d-93bc-cf28ebbb0f9c" />

### 3D Label Reconstruction

This module converts one selected non-background label into a surface mesh. The selected object is cropped before mesh extraction to reduce memory use.

| Control | Description |
|---|---|
| `Z Ratio` | Scaling factor applied to the mesh along Z. Use the physical Z-to-XY spacing ratio. |
| `Reconstruct Selected Label` | Reconstruct the currently selected label and display it as a `Surface` layer in a separate 3D napari viewer. |
| `Load Mesh` | Load a previously saved `.npz` mesh. |
| `Save Mesh` | Save vertices, faces, voxel count, Z ratio, and label ID in a compressed `.npz` file. Enabled after a mesh has been generated. |

### Full 3D View

| Control | Description |
|---|---|
| `Z Ratio` | Set layer scale to `(Z Ratio, 1.0, 1.0)` for data ordered as `(Z, Y, X)`. |
| `Switch to 3D View` | Switch the main viewer from slice mode to volume rendering and pan/zoom interaction. |
| `Switch to 2D View` | Return to two-dimensional slice navigation. |

### Morphometry Analysis

Morphometry tasks support three-dimensional label layers. Raw image data are needed only for intensity-based features.

#### Common settings

| Control | Description |
|---|---|
| `Voxel size (Z / Y / X)` | Physical voxel dimensions used for volumes, areas, centroids, distances, and shape measurements. |
| `Output folder` | Root output directory. Each task writes to a task-specific subfolder. |
| `Browse` | Select the root output directory. |
| `Open Output Folder` | Open the current analysis output directory. |

#### 1. Basic Information

Compute per-cell morphology and optional intensity measurements.

| Control | Description |
|---|---|
| `Raw layer` | Raw image used for intensity measurements. It is not required when no intensity feature is selected. |
| `Refresh Raw Layers` | Rebuild the raw-layer selector from image layers currently loaded in napari. |
| `Features to compute` | Select one or more supported measurements. |
| `Run Basic Information` | Calculate features and save cell-level and summary CSV files. |
| `Show feature` | Display a supported scalar feature as a label-mapped 3D image. |

Supported features include:

| Feature | Description |
|---|---|
| Volume | Physical cell volume calculated from voxel count and voxel size. |
| Equivalent diameter | Diameter of a sphere with the same volume. |
| Surface area | Surface area estimated from exposed voxel faces. |
| Centroid | Cell centroid in physical coordinates. |
| Sphericity | Shape compactness relative to a sphere. |
| Compactness | Surface-area-to-volume compactness measurement. |
| Axis major | Major-axis length estimated from PCA of physical voxel coordinates. |
| Elongation | Ratio representing long-axis elongation. |
| Flatness | Ratio representing compression along the minor axis. |
| Min / Max / Mean / Std intensity | Intensity statistics within each label. A raw image layer is required. |

Typical outputs:

```text
basic_info/
├── basic_info_cell_features.csv
└── basic_info_summary.csv
```

#### 2. Neighborhood Analysis

Calculate centroid-based neighborhoods.

| Control | Description |
|---|---|
| `Mode` | Select `kNN distance` or `Radius count / density`. |
| `k` | Number of neighbors used in k-nearest-neighbor mode. |
| `Radius pixels` | Radius used in radius mode, expressed in voxel/pixel coordinates as defined by the current implementation. |
| `Run Neighborhood` | Calculate neighborhood features and summary statistics. |
| `Show feature` | Map a supported neighborhood feature back to the label volume. |
| `Run Local Comparison` | Compare each cell with its local neighbors and calculate a local z-score for the selected feature. |

In `kNN distance` mode, outputs can include nearest-neighbor distance, mean and median kNN distance, and local density. In radius mode, outputs can include neighbor count and local density.

Typical outputs:

```text
neighborhood/
├── neighborhood_cell_features.csv
└── neighborhood_summary.csv
```

Local comparison records the original feature value, local mean, local standard deviation, and local z-score.

#### 3. Contact Graph Analysis

Detect face-touching relationships between labeled cells.

| Control | Description |
|---|---|
| `Features to compute` | Select neighbor count, total contact area, mean contact area, maximum contact area, and/or contact fraction. |
| `Run Contact Graph` | Calculate pairwise contact edges and per-cell contact summaries. |
| `Show feature` | Map a supported contact feature back to the label volume. |

Contact area is estimated from shared voxel faces using the physical voxel size.

Typical outputs:

```text
contact/
├── contact_cell_features.csv
├── contact_edges.csv
└── contact_summary.csv
```

#### 4. Clustering

Cluster cells using one selected scalar feature.

| Control | Description |
|---|---|
| `Feature` | Select a morphology, intensity, neighborhood, or contact-related feature available to the current analysis workflow. |
| `Clusters` | Number of K-means clusters. |
| `Run Clustering` | Fit K-means to valid feature values, save the assignments, and display `cluster_id` in napari. |

A raw image layer is required when the selected clustering feature is intensity-based.

Typical outputs:

```text
clustering/
├── clustering_cell_features.csv
└── clustering_summary.csv
```

## Keyboard Shortcuts

| Shortcut | Context | Action |
|---|---|---|
| `Q` | Viewer | Move to the previous Z slice. |
| `W` | Viewer | Move to the next Z slice. |
| `A` | Manual curation | Start `Add to Label`. |
| `S` | Manual curation | Start `Subtract from Label`. |
| `Ctrl + D` | Manual curation | Start `Add New Label`. |
| `C` | Manual curation | Change the selected label ID. |
| `Ctrl + A` | ROI editing | Apply the current ROI. |
| `Ctrl + C` | ROI editing | Cancel the current ROI. |
| `Delete` | Manual curation | Delete the selected label on the current Z slice. |
| `Ctrl + Delete` | Manual curation | Delete the selected label across all Z slices. |
| `+` / `-` | Label or ROI brush | Increase or decrease brush size. |
| `Ctrl + Z` | One-click segmentation | Undo the latest local refinement. |

Keyboard shortcuts may be intercepted by the currently focused text field. Click the viewer before using a shortcut when it does not respond.

## Output Files and Folders

### Segmentation run

A typical output directory contains:

```text
output_<image>_<timestamp>/
├── curation_labels.zarr
├── <instance-map output from backend>
├── <confidence-map output, when enabled>
├── <inference log, when produced>
└── curated_patch/
```

Exact backend filenames can vary, but `curation_labels.zarr` is the editable label store used by the plugin after inference.

### Curated training patches

```text
curated_patch/
├── imagesTr/
│   ├── patch_0001.tif
│   └── ...
└── labelsTr/
    ├── patch_0001.tif
    └── ...
```

Image and label filenames must remain paired.

### Morphometry results

Each analysis task writes into its own subfolder under the selected output root:

```text
<analysis_output>/
├── basic_info/
├── neighborhood/
├── contact/
└── clustering/
```

## Notes on Large Volumes

- Prefer Zarr for label loading, empty-label creation, curation, and saving.
- Keep only the required image and label layers visible when interactive rendering becomes slow.
- Use a larger inference stride to reduce overlap and runtime, recognizing that this may reduce boundary redundancy.
- Increase batch size only after checking available GPU memory.
- Save curated labels before starting a separate analysis or patch-curation session.
