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

#### 1. Create a new environment

```bash
conda create -n focus3d python=3.10 -y
conda activate focus3d
```

#### 2. Install torch
For CUDA 12.x, replace cu12x with your specific CUDA-compatible PyTorch build. For example, for CUDA 12.6:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
Please check the official PyTorch installation guide if you need another CUDA version.

#### 3. Install FOCUS-3D

```bash
pip install -U "focus-3d[gui]"
```

#### 4. Install detectron2 (only for Linux model fine-tuning)
For Linux, FOCUS-3D requires Detectron2 for segmentation model fine-tuning. Please install Detectron2 according to the official guide:

https://detectron2.readthedocs.io/en/latest/tutorials/install.html

For most Linux environments, the official source installation command is:

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

#### 5. Download pretrained model
Users can download the pretrained model for 3D segmentation: https://huggingface.co/Qinghua-thu/FOCUS-3D/.


#### 6. Launch napari

```bash
python -m napari
```

## Recommended Workflow

### Step 1 — Load and inspect the image
1. In napari, open `Plugins -> 3D Segmentation (FOCUS-3D)`.
2. Load a raw 3D microscopy image:
   - use `File -> Open Folder` for a `.zarr` dataset, which is recommended for large volumes;
   - use `File -> Open File(s)` for `.tif` or `.tiff` images.
3. Open the `Basic` tab and use `Display Settings` when the raw image is difficult to inspect:
   - move the `Minimum` and `Maximum` sliders to adjust contrast;
   - click `Auto` for percentile-based contrast adjustment;

### Step 2 — Run automatic 3D segmentation
1. Open the `Segmentation` tab.
2. In `Run Segmentation`, set the parameters that are most likely to vary between datasets:
   - **Z Ratio** — the physical Z-to-XY spacing ratio. Use `1.0` for isotropic data.
   - **Output Path** — the directory used for the segmentation result.
   - **Checkpoint** — the pretrained or fine-tuned checkpoint.
   - **Cell radius (pixel)** — the approximate cell radius in the XY plane.
   - **Background intensity** — patches or cells with grayscale values less than this value will be removed.
   - **Min size (3D)** and **Max size (3D)** — remove small or large instances.
3. Use `Advanced` only when you need to change the GPU, configuration file, normalization percentiles, patch stride, batch size, or stitching thresholds. See the [complete menu reference](docs/MENU_REFERENCE.md#run-segmentation) for parameter definitions and defaults.
4. Click `Run 3D Segmentation`.

After inference, FOCUS-3D loads a label layer into napari. The inference outputs are saved to the specified output path in both TIFF and Zarr formats.

### Step 3 — Inspect and curate the segmentation

1. Return to the `Basic` tab.

2. Click `Enter Curation Mode`, then click a cell label to select it.

3. Correct common errors:

   * use `Add to Label` to recover missing regions;
   * use `Subtract from Label` to remove incorrect regions;
   * use `Add New Label` for a missed cell;
   * use `Delete Current Z` for a slice-specific error;
   * use `Delete All Z` to remove an incorrect 3D instance;
   * use `Delete Inside ROI (All Z)` to remove multiple labels in a selected region.

   For detailed instructions on label-editing operations and keyboard shortcuts, see [Manual Curation](docs/MENU_REFERENCE.md#manual-curation).

4. For labels stored in Zarr format, edits are written directly to the underlying Zarr data, so no separate save step is required. Labels loaded from TIFF are edited in memory and must be saved manually from the `Save` panel after curation.

### Step 4 — Use one-click segmentation when needed

One-click segmentation can accelerate the curation.

1. Keep both the raw image and segmentation label layer loaded.
2. Open `Segmentation -> One-click segmentation`.
3. Click `Enter Inactive Mode` to load the local refinement model and activate interactive refinement.
4. Click the target cell in the viewer and inspect the updated label.
5. Click `Exit Inactive Mode` after finishing.


### Step 5 — Analyze the segmentation results

Open the `Analysis` tab after the segmentation has been checked.

#### Reconstruct one selected cell

1. Select a non-background cell in the label layer.
2. Set the `Z Ratio`.
3. Click `Reconstruct Selected Label`.
4. Save the reconstructed mesh as `.npz` when needed.

#### Inspect the full volume in 3D

1. Set the physical `Z Ratio`.
2. Click `Switch to 3D View`.
3. Rotate and inspect the image and labels.
4. Click `Switch to 2D View` to return to slice navigation.

#### Run morphometry analysis

1. Set the physical voxel size in Z, Y, and X.
2. Choose an output folder.
3. Run one or more tasks:
   - `Basic Information` for cell morphology and optional intensity measurements;
   - `Neighborhood Analysis` for centroid-based local organization;
   - `Contact Graph Analysis` for face-touching cell relationships;
   - `Clustering` for feature-based cell grouping.
4. Use `Show feature` to map supported results back to the napari label volume.

### Step 6 — Prepare curated patches and fine-tune the model

Fine-tuning is a two-stage workflow: curate training patches in napari, then run the training notebook.

#### A. Export curated patches from napari

1. Keep the raw image and corrected label volume loaded.
2. Open `Segmentation -> Finetune with Current Labels`.
3. Click `Calculate Valid Patches`.
4. Inspect the patch boxes and choose a `Patch ID`.
5. Set the patch `Save Path`.
6. Click `Curate Selected Patch`.
7. In the new patch viewer, correct the labels with the `Basic` curation tools.
8. In `Save Curated Patch`, click `Save`.

Each saved sample is written as a paired TIFF image and label:

```text
<save_path>/
├── imagesTr/
│   ├── patch_0001.tif
│   └── ...
└── labelsTr/
    ├── patch_0001.tif
    └── ...
```

Use `Clear Patch Boxes` when you want to remove the patch overlays and return to normal curation.

#### B. Run fine-tuning from the notebook

1. Expand the collapsed `Fine-tune` instruction inside the same panel.
2. Open:

```text
notebooks/02_finetune.ipynb
```

3. Configure the notebook to use the curated patch directory.
4. Run fine-tuning and obtain a new checkpoint.
5. Return to `Segmentation -> Run Segmentation`.
6. Select the new checkpoint in the `Checkpoint` field and run segmentation again.

The napari panel prepares and exports training data, but it does not launch model training directly.

## Detailed Interface Reference

The complete descriptions of all controls are maintained in:

- [Basic tab](docs/MENU_REFERENCE.md#basic-tab)
- [Segmentation tab](docs/MENU_REFERENCE.md#segmentation-tab)
- [Analysis tab](docs/MENU_REFERENCE.md#analysis-tab)
- [Keyboard shortcuts](docs/MENU_REFERENCE.md#keyboard-shortcuts)
- [Output files and folders](docs/MENU_REFERENCE.md#output-files-and-folders)


## Issues

If you encounter a problem, please [file an issue](https://github.com/Qinghua24/cellseg/issues) with a detailed description, relevant logs, and a minimal example when possible. You can also contact `zhangqh24@mails.tsinghua.edu.cn`.

## Citing

Please contact us before the paper is published.

