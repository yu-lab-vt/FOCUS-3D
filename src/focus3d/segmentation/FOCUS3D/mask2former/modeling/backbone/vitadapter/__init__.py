import os


def _is_windows_backend():
    backend = os.environ.get('CELLSEG_FOCUS3D_BACKEND', 'auto').strip().lower()

    if backend in {'windows', 'win', 'nod2', 'no_detectron2', 'pytorch'}:
        return True

    if backend in {'detectron2', 'd2', 'linux'}:
        return False

    return os.name == 'nt'


if _is_windows_backend():
    from .vitadapter_win import ViTAdapter
else:
    from .vitadapter import ViTAdapter
