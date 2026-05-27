from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Dict, Union


def _emit_status(progress_callback, message: str) -> None:
    if progress_callback is None:
        return
    progress_callback(-1, str(message))


def _check_cancelled(cancel_callback) -> None:
    if cancel_callback is not None and cancel_callback():
        raise RuntimeError('__FINETUNE_CANCELLED__')


def _resolve_path(
    path: Union[str, Path] | None, base_dir: Union[str, Path] | None = None
):
    if path is None:
        return None

    p = Path(str(path)).expanduser()
    if p.is_absolute():
        return p

    if base_dir is not None:
        p_base = Path(base_dir) / p
        if p_base.exists():
            return p_base.resolve()

    return p.resolve()


def _try_defrost(cfg):
    if hasattr(cfg, 'defrost'):
        cfg.defrost()


def _try_freeze(cfg):
    if hasattr(cfg, 'freeze'):
        cfg.freeze()


def _try_set_attr(obj, name: str, value) -> bool:
    if hasattr(obj, name):
        setattr(obj, name, value)
        return True
    return False


def _try_set_cfg_value(cfg, dotted_key: str, value) -> bool:
    """
    Safely set cfg.A.B.C = value only if the path already exists.
    This avoids breaking configs that do not define the key.
    """
    parts = dotted_key.split('.')
    node = cfg

    for part in parts[:-1]:
        if not hasattr(node, part):
            return False
        node = getattr(node, part)

    if not hasattr(node, parts[-1]):
        return False

    setattr(node, parts[-1], value)
    return True


def _find_checkpoint(output_dir: Path) -> str | None:
    """
    Return a likely final checkpoint path after training.
    """
    candidates = [
        output_dir / 'model_final.pth',
        output_dir / 'model_best.pth',
        output_dir / 'checkpoint.pth',
    ]

    for p in candidates:
        if p.exists():
            return str(p)

    # Fallback: newest .pth file.
    pths = sorted(
        output_dir.glob('*.pth'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pths:
        return str(pths[0])

    return None


def run_finetune(
    *,
    config_file: Union[str, Path],
    curated_patch_dir: Union[str, Path],
    output_dir: Union[str, Path],
    init_checkpoint: Union[str, Path] | None = None,
    cuda_visible_devices: str | None = None,
    resume: bool = True,
    opts: Sequence[str] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> Dict[str, Any]:
    """
    UI-callable fine-tuning entry.

    Parameters
    ----------
    config_file:
        Path to FOCUS3D config yaml.

    curated_patch_dir:
        Directory containing curated training patches.

    output_dir:
        Directory where fine-tuned checkpoints will be written.

    init_checkpoint:
        Initial checkpoint. Usually the current checkpoint shown in UI.

    cuda_visible_devices:
        Optional GPU id string, e.g. "0".

    resume:
        Passed to run_train(cfg, resume=resume).

    opts:
        Optional additional config overrides, same style as command-line opts:
        ["SOLVER.BASE_LR", "1e-5", "SOLVER.MAX_ITER", "1000"]

    Returns
    -------
    dict with output_dir and final_checkpoint.
    """
    t0 = time.time()

    _emit_status(progress_callback, 'Preparing fine-tuning...')
    _check_cancelled(cancel_callback)

    if (
        cuda_visible_devices is not None
        and str(cuda_visible_devices).strip() != ''
    ):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices).strip()

    focus3d_dir = Path(__file__).resolve().parent

    config_file = _resolve_path(config_file, base_dir=focus3d_dir)
    curated_patch_dir = _resolve_path(curated_patch_dir)
    output_dir = _resolve_path(output_dir)

    if config_file is None or not Path(config_file).exists():
        raise FileNotFoundError(f'Config file does not exist: {config_file}')

    if curated_patch_dir is None:
        raise ValueError('curated_patch_dir is empty.')

    curated_patch_dir = Path(curated_patch_dir)
    if not curated_patch_dir.exists():
        raise FileNotFoundError(
            f'Curated patch directory does not exist:\n{curated_patch_dir}\n\n'
            'Please export/save current labels to curated_patch before fine-tuning.'
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_checkpoint = (
        _resolve_path(init_checkpoint) if init_checkpoint else None
    )

    # These env vars are useful if your train_net_3d.py / dataset registration
    # reads paths from environment.
    os.environ['FOCUS3D_CURATED_PATCH_DIR'] = str(curated_patch_dir)
    os.environ['FOCUS3D_FINETUNE_OUTPUT_DIR'] = str(output_dir)

    if init_checkpoint is not None:
        os.environ['FOCUS3D_INIT_CHECKPOINT'] = str(init_checkpoint)

    _emit_status(progress_callback, 'Loading training backend...')
    _check_cancelled(cancel_callback)

    # Import after CUDA_VISIBLE_DEVICES is set.
    from train_net_3d import build_cfg, run_train

    cfg_opts = []
    if opts:
        cfg_opts.extend([str(x) for x in opts])

    # Common Detectron2-style overrides.
    cfg_opts.extend(['OUTPUT_DIR', str(output_dir)])

    if init_checkpoint is not None:
        cfg_opts.extend(['MODEL.WEIGHTS', str(init_checkpoint)])

    _emit_status(progress_callback, 'Building fine-tuning config...')
    _check_cancelled(cancel_callback)

    cfg = build_cfg(str(config_file), cfg_opts)

    _try_defrost(cfg)

    # Set common fields if they exist.
    _try_set_cfg_value(cfg, 'OUTPUT_DIR', str(output_dir))

    if init_checkpoint is not None:
        _try_set_cfg_value(cfg, 'MODEL.WEIGHTS', str(init_checkpoint))

    # These are optional. They will only be set if your cfg already defines them.
    # If not, train_net_3d.py should read FOCUS3D_CURATED_PATCH_DIR from env.
    _try_set_cfg_value(
        cfg, 'DATASETS.CURATED_PATCH_DIR', str(curated_patch_dir)
    )
    _try_set_cfg_value(cfg, 'DATASETS.TRAIN_DIR', str(curated_patch_dir))
    _try_set_cfg_value(cfg, 'INPUT.CURATED_PATCH_DIR', str(curated_patch_dir))
    _try_set_cfg_value(
        cfg, 'FOCUS3D.CURATED_PATCH_DIR', str(curated_patch_dir)
    )

    _try_freeze(cfg)

    meta = {
        'config_file': str(config_file),
        'curated_patch_dir': str(curated_patch_dir),
        'output_dir': str(output_dir),
        'init_checkpoint': None
        if init_checkpoint is None
        else str(init_checkpoint),
        'resume': bool(resume),
        'opts': cfg_opts,
    }

    with open(
        output_dir / 'finetune_ui_args.json', 'w', encoding='utf-8'
    ) as f:
        json.dump(meta, f, indent=2)

    _emit_status(progress_callback, 'Running fine-tuning...')
    _check_cancelled(cancel_callback)

    train_result = run_train(cfg, resume=resume)

    _emit_status(
        progress_callback, 'Fine-tuning finished. Searching checkpoint...'
    )
    _check_cancelled(cancel_callback)

    final_checkpoint = _find_checkpoint(output_dir)

    result = {
        'output_dir': str(output_dir),
        'final_checkpoint': final_checkpoint,
        'train_result': train_result,
        'time_sec': time.time() - t0,
    }

    with open(output_dir / 'finetune_result.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'output_dir': result['output_dir'],
                'final_checkpoint': result['final_checkpoint'],
                'time_sec': result['time_sec'],
            },
            f,
            indent=2,
        )

    _emit_status(progress_callback, 'Fine-tuning completed.')

    return result
