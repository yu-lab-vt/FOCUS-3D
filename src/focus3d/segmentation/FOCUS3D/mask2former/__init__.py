"""
FOCUS-3D Mask2Former package.

This package intentionally performs no eager backend imports.

Backend-specific entry points must explicitly import either:
- config_win.py / maskformer_model_win.py for pure-PyTorch inference
- config.py / maskformer_model.py / modeling for Detectron2 training
"""
