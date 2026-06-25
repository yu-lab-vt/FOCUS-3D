from __future__ import annotations

import importlib.util
import inspect
import os
import platform
import sys
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

MASK2FORMER_DIR = Path(__file__).resolve().parent / 'FOCUS3D'

# Linux / original Detectron2 version
INFERENCE_PY_DETECTRON2 = MASK2FORMER_DIR / 'inference.py'

# Windows / no-Detectron2 version
INFERENCE_PY_WINDOWS = MASK2FORMER_DIR / 'inference_win.py'

# Cache loaded inference module by backend.
_INFERENCE_MODULE_CACHE = {}


def _normalize_cuda_visible_devices(cuda_visible_devices=None):
    """
    Normalize CUDA_VISIBLE_DEVICES-like text from UI.

    Examples:
        None -> None
        '' -> None
        '0' -> '0'
        '1' -> '1'
        '0,1' -> '0,1'
    """
    if cuda_visible_devices is None:
        return None

    text = str(cuda_visible_devices).strip()
    if text == '':
        return None

    text = text.replace(' ', '')
    text = text.replace(';', ',')

    ids = [x for x in text.split(',') if x != '']
    if len(ids) == 0:
        return None

    return ','.join(ids)


def _normalize_torch_device(device=None, cuda_visible_devices=None):
    """
    Resolve device to an explicit torch device string.

    Good:
        cuda:0
        cuda:1
        cpu

    Avoid:
        cuda
        None
    """
    import torch

    if not torch.cuda.is_available():
        return 'cpu'

    cuda_visible_devices = _normalize_cuda_visible_devices(
        cuda_visible_devices
    )

    if device is None or str(device).strip() == '':
        if cuda_visible_devices:
            first = cuda_visible_devices.split(',')[0].strip()
            device = f'cuda:{int(first)}' if first.isdigit() else 'cuda:0'
        else:
            device = 'cuda:0'
    else:
        device = str(device).strip()

    if device == 'cuda':
        device = 'cuda:0'

    torch_device = torch.device(device)

    if torch_device.type != 'cuda':
        return str(torch_device)

    dev_index = torch_device.index
    if dev_index is None:
        dev_index = 0

    n_cuda = torch.cuda.device_count()
    if dev_index < 0 or dev_index >= n_cuda:
        raise RuntimeError(
            f'Requested device cuda:{dev_index}, but PyTorch only sees '
            f'{n_cuda} CUDA device(s). '
            f'Please check the GPU selector and CUDA_VISIBLE_DEVICES.'
        )

    return f'cuda:{dev_index}'


def _activate_torch_device(device):
    """
    Make selected CUDA device current before model construction.
    """
    import torch

    device = _normalize_torch_device(device)

    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch_device = torch.device(device)
        torch.cuda.set_device(torch_device)

        print(
            f'[FOCUS3D adapter] activated torch device: {device}',
            flush=True,
        )
        print(
            f'[FOCUS3D adapter] current cuda device: cuda:{torch.cuda.current_device()}',
            flush=True,
        )
        print(
            f'[FOCUS3D adapter] GPU name: {torch.cuda.get_device_name(torch_device.index)}',
            flush=True,
        )
    else:
        print('[FOCUS3D adapter] activated torch device: cpu', flush=True)

    return device


def _resolve_relative_to_mask2former(path: str | Path) -> str:
    """
    Resolve config / checkpoint path.

    Priority:
    1. Absolute path
    2. Relative to Mask2former folder
    3. Relative to current working directory
    """
    p = Path(str(path)).expanduser()

    if p.is_absolute():
        return str(p)

    p_mask2former = MASK2FORMER_DIR / p
    if p_mask2former.exists():
        return str(p_mask2former)

    p_cwd = Path.cwd() / p
    if p_cwd.exists():
        return str(p_cwd.resolve())

    # Default: interpret relative paths like configs/3d_test.yaml
    # relative to Mask2former.
    return str(p_mask2former)


def _select_inference_file():
    """
    Select inference backend.

    Default behavior:
    - Windows: use inference_win.py
    - Linux/macOS: use inference.py

    Optional override:
    set CELLSEG_FOCUS3D_BACKEND to:
        - "windows", "nod2", "no_detectron2", "pytorch"
        - "detectron2", "d2", "linux"
    """
    backend = os.environ.get('CELLSEG_FOCUS3D_BACKEND', 'auto').strip().lower()

    if backend in {'windows', 'win', 'nod2', 'no_detectron2', 'pytorch'}:
        return INFERENCE_PY_WINDOWS, 'windows'

    if backend in {'detectron2', 'd2', 'linux'}:
        return INFERENCE_PY_DETECTRON2, 'detectron2'

    # Auto mode
    system_name = platform.system().lower()
    if system_name.startswith('win') or os.name == 'nt':
        return INFERENCE_PY_WINDOWS, 'windows'

    return INFERENCE_PY_DETECTRON2, 'detectron2'


def _resolve_output_dir(path: str | Path) -> str:
    """
    Resolve output dir before changing cwd.
    """
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())


def _resolve_image_path(path: str | Path) -> str | None:
    if path is None:
        return None

    text = str(path).strip()
    if not text:
        return None

    p = Path(text).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p

    if p.exists() and p.suffix.lower() in ['.tif', '.tiff', '.zarr']:
        return str(p.resolve())

    return None


def _load_mask2former_inference_module():
    """
    Dynamically load FOCUS3D inference module.

    Backend selection:
    - Linux / Detectron2: FOCUS3D/inference.py
    - Windows / no Detectron2: FOCUS3D/inference_win.py
    """
    inference_py, backend_name = _select_inference_file()

    cache_key = (
        str(inference_py.resolve())
        if inference_py.exists()
        else str(inference_py)
    )
    if cache_key in _INFERENCE_MODULE_CACHE:
        return _INFERENCE_MODULE_CACHE[cache_key]

    if not inference_py.exists():
        raise FileNotFoundError(
            f"Cannot find FOCUS3D inference file for backend '{backend_name}':\n"
            f'{inference_py}\n\n'
            f'Expected files:\n'
            f'  Detectron2 backend: {INFERENCE_PY_DETECTRON2}\n'
            f'  Windows backend:    {INFERENCE_PY_WINDOWS}\n'
        )

    # Needed because inference.py / inference_win.py may import local modules
    # from the FOCUS3D folder.
    mask2former_dir_str = str(MASK2FORMER_DIR)
    if mask2former_dir_str not in sys.path:
        sys.path.insert(0, mask2former_dir_str)

    module_name = f'_cellseg_focus3d_inference_{backend_name}'

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(inference_py),
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f'Failed to load inference module from {inference_py}'
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, 'infer_volume'):
        raise AttributeError(f'{inference_py} does not define infer_volume().')

    _INFERENCE_MODULE_CACHE[cache_key] = module
    return module


@contextmanager
def _temporarily_chdir(path: Path):
    """
    Some Mask2former code may assume relative paths such as configs/xxx.yaml.
    We temporarily set cwd to Mask2former folder during inference.
    """
    old_cwd = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _image_data_to_temp_tif(
    image_data,
    output_dir: str | Path,
    input_name: str = 'napari_layer',
) -> str:
    """
    Save current napari layer data to a temporary tif, because infer_volume()
    currently expects image_path.
    """
    if image_data is None:
        raise ValueError(
            'image_data is None and no valid image_path was provided.'
        )

    if hasattr(image_data, 'compute'):
        arr = image_data.compute()
    else:
        arr = np.asarray(image_data)

    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(
            f'Expected 3D image data after squeeze, got shape {arr.shape}'
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = ''.join(
        c if c.isalnum() or c in '-_.' else '_' for c in str(input_name)
    )
    tmp_path = output_dir / f'__tmp_{safe_name}_{uuid.uuid4().hex}.tif'

    tifffile.imwrite(tmp_path, arr.astype(np.float32, copy=False))
    return str(tmp_path)


def run_mask2former_inference(
    *,
    image_data=None,
    image_path: str | Path | None = None,
    input_name: str = 'napari_layer',
    config_file: str | Path,
    weights_path: str | Path,
    output_dir: str | Path,
    cuda_visible_devices: str | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> dict:
    """
    Stable UI-side wrapper for Mask2Former infer_volume().

    The UI calls this function.
    This function calls segmentation/FOCUS3D/inference.py::infer_volume().
    """
    # Backward compatibility:
    # If an older caller passes device through kwargs, consume it here.
    if device is None:
        device = kwargs.pop('device', None)
    else:
        kwargs.pop('device', None)

    cuda_visible_devices = _normalize_cuda_visible_devices(
        cuda_visible_devices
    )

    device = _normalize_torch_device(
        device=device,
        cuda_visible_devices=cuda_visible_devices,
    )

    # Important:
    # Do NOT rely on changing CUDA_VISIBLE_DEVICES inside a running napari process.
    # PyTorch may already be imported/initialized. The reliable control path is:
    # torch.cuda.set_device + infer_volume(device='cuda:N').
    #
    # Keep cuda_visible_devices as metadata only. For subprocess training it can
    # still be useful, but for in-process inference use explicit torch device.
    device = _activate_torch_device(device)

    print(
        f'[FOCUS3D adapter] run_mask2former_inference: '
        f'CUDA_VISIBLE_DEVICES={cuda_visible_devices}, device={device}',
        flush=True,
    )

    output_dir_abs = _resolve_output_dir(output_dir)
    config_file_abs = _resolve_relative_to_mask2former(config_file)
    weights_path_abs = _resolve_relative_to_mask2former(weights_path)

    source_image_path = _resolve_image_path(image_path)
    tmp_image_path = None

    if source_image_path is None:
        tmp_image_path = _image_data_to_temp_tif(
            image_data=image_data,
            output_dir=output_dir_abs,
            input_name=input_name,
        )
        source_image_path = tmp_image_path

    module = _load_mask2former_inference_module()
    infer_volume = module.infer_volume

    call_kwargs = dict(
        image_path=source_image_path,
        config_file=config_file_abs,
        weights_path=weights_path_abs,
        output_dir=output_dir_abs,
        device=device,
        **kwargs,
    )

    # Future-proof:
    # only pass arguments supported by the current infer_volume().
    signature = inspect.signature(infer_volume)
    supported = set(signature.parameters.keys())
    call_kwargs = {k: v for k, v in call_kwargs.items() if k in supported}

    print(
        f'[FOCUS3D adapter] infer_volume device argument: '
        f'{call_kwargs.get("device")}',
        flush=True,
    )

    try:
        with _temporarily_chdir(MASK2FORMER_DIR):
            result = infer_volume(**call_kwargs)

        if isinstance(result, dict):
            result.setdefault('device', device)
            result.setdefault('cuda_visible_devices', cuda_visible_devices)

        return result

    finally:
        if tmp_image_path is not None:
            with suppress(Exception):
                Path(tmp_image_path).unlink(missing_ok=True)
