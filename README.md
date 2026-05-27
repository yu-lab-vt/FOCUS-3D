# cellseg

[![License BSD-3](https://img.shields.io/pypi/l/cellseg.svg?color=green)](https://github.com/Qinghua24/cellseg/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cellseg.svg?color=green)](https://pypi.org/project/cellseg)
[![Python Version](https://img.shields.io/pypi/pyversions/cellseg.svg?color=green)](https://python.org)
[![tests](https://github.com/Qinghua24/cellseg/workflows/tests/badge.svg)](https://github.com/Qinghua24/cellseg/actions)
[![codecov](https://codecov.io/gh/Qinghua24/cellseg/branch/main/graph/badge.svg)](https://codecov.io/gh/Qinghua24/cellseg)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/cellseg)](https://napari-hub.org/plugins/cellseg)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

CellSeg is a user-friendly napari plugin for 3D cell segmentation. Cellseg supports a series of functions such as 3D automatic segmentation, manual curation, 3D structure reconstruction, fine-tuning, quantitative analysis, etc.
<p align="center">
<img width="823" height="492" alt="image" src="https://github.com/user-attachments/assets/c45dd0ef-73e3-48bd-849a-915669cf2418" />
  <br>
</p>

## Installation

You can install `cellseg` with the following steps:

1. Create a new environment in the directory containing `environment.yml`:

   ```bash
   cd your path
   conda env create -n cellseg -f environment.yml
   ```

2. Install `cellseg`:

   ```bash
   conda activate cellseg
   pip install napari[all]
   pip install -e .
   ```

3. Launch the application:

   ```bash
   napari
   ```

## Usage
1. Open CellSeg from `Plugins -> 3D Segmentation (CellSeg)`.
2. Open the raw image from `File -> Open Folder` for a `.zarr` dataset (recommended), or `File -> Open File(s)` for `.tif` images.
3. Load labels from `Load Label -> Load from Zarr` (recommended) or `Load Label -> Load from TIFF`.
4. Manual curation: go to `Manual Curation -> Enter Curation Mode`.
   For `.zarr` labels, curation results are written directly to the file and saved automatically.

| Operation | Shortcut | Description |
|---|---|---|
| Enter Curation Mode | — | Activate label editing mode so you can select and modify labels manually. |
| Polygon Draw Mode | — | Define an ROI by clicking polygon vertices; the enclosed region is used as the selected area. |
| Brush Draw Mode | — | Define an ROI by painting directly on the image; the painted region is used as the selected area. |
| Add to Label | `A` | Add the selected ROI to the current label. |
| Subtract from Label | `S` | Remove the selected ROI from the current label. |
| Add New Label | `Ctrl + D` | Create a new label. Regions from multiple Z slices can be combined into the same label. |
| Change Label | `C` | Change the ID of the currently selected label. |
| Apply ROI | `Ctrl + A` | Apply the current ROI operation. Used with **Add to Label**, **Subtract from Label**, and **Add New Label**. |
| Cancel ROI | `Ctrl + C` | Cancel the current ROI operation without applying changes. |
| Delete Current Z | `Delete` | Delete the selected label only in the current Z slice. |
| Delete All Z | `Ctrl + Delete` | Delete the selected label across all Z slices. |
| Delete Inside ROI (All Z) | — | Delete all labels inside the selected ROI across all Z slices. |
| Export Log | — | Export curation actions and label IDs to an Excel file. |

5. Save the curation result as `.zarr` or `.tif`; label IDs can also be reassigned if needed.
6. 3D reconstruction: reconstruct the selected label, adjust the Z ratio, and save or load the mesh.
 <p align="center">
   <img width="625" height="422" alt="image" src="https://github.com/user-attachments/assets/15616b42-5524-431a-a619-ebb6810c7181" />
  <br>
</p>

7. Calculate Size Distribution: compute cell count and cell size statistics.
<p align="center">
  <img width="520" height="236" alt="image" src="https://github.com/user-attachments/assets/da6891eb-765a-43e2-ba90-e155528e497c" />
  <br>
</p>

## License

Distributed under the terms of the [BSD-3] license, "cellseg" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description or contact zhangqh24@mails.tsinghua.edu.cn.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
