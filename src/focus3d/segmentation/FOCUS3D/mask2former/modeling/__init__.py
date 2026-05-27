import os


def _is_windows_backend():
    backend = os.environ.get('CELLSEG_FOCUS3D_BACKEND', 'auto').strip().lower()

    if backend in {'windows', 'win', 'nod2', 'no_detectron2', 'pytorch'}:
        return True

    if backend in {'detectron2', 'd2', 'linux'}:
        return False

    return os.name == 'nt'


if not _is_windows_backend():
    # Linux / Detectron2 backend.
    # Keep original registration imports here.
    from .backbone.mae3d_backbone import D2MAE3DBackbone
else:
    # Windows / no-Detectron2 backend.
    # Important: do not import original modeling modules here.
    pass
