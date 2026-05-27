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
    from .mask2former_transformer_decoder_3d import (
        MultiScaleMaskedTransformerDecoder,
    )
else:
    # Windows / no-Detectron2 backend.
    # Do not import the original Detectron2-based decoder here.
    pass
