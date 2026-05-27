import os

_BACKEND = os.environ.get('CELLSEG_FOCUS3D_BACKEND', 'auto').strip().lower()
_IS_WINDOWS_BACKEND = _BACKEND in {
    'windows',
    'win',
    'nod2',
    'no_detectron2',
    'pytorch',
} or (_BACKEND == 'auto' and os.name == 'nt')

if not _IS_WINDOWS_BACKEND:
    # Linux / Detectron2 path: keep original registration behavior.
    from . import modeling
    from .config import add_maskformer2_config
else:
    # Windows path: do not import Detectron2 registration modules.
    def add_maskformer2_config(*args, **kwargs):
        raise RuntimeError(
            'add_maskformer2_config is Detectron2-only and should not be used '
            'by the Windows no-Detectron2 inference path.'
        )
