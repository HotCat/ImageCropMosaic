# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

project_root = Path(SPECPATH)
sam2_root = project_root / "sam2"

excludes = [
    'tkinter',
    'matplotlib',
    'IPython',
    'jupyter',
    'notebook',
    'pytest',
    'sphinx',
    'docutils',
    'unittest',
    'test',
    'tests',
]

a = Analysis(
    ['main.py'],
    pathex=[
        str(project_root),
        str(sam2_root),
    ],
    binaries=[],
    datas=[
        (str(sam2_root / 'sam2' / 'configs'), 'sam2/configs'),
        (str(sam2_root / 'checkpoints'), 'sam2/checkpoints'),
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torch.backends',
        'torch.backends.mps',
        'numpy',
        'PIL',
        'cv2',
        'sam2',
        'sam2.build_sam',
        'sam2.sam2_image_predictor',
        'sam2.modeling',
        'sam2.modeling.sam2_base',
        'sam2.modeling.sam',
        'sam2.modeling.sam.prompt_encoder',
        'sam2.modeling.sam.mask_decoder',
        'sam2.modeling.backbones',
        'sam2.modeling.backbones.image_encoder',
        'sam2.utils',
        'sam2.utils.transforms',
        'hydra',
        'hydra.utils',
        'hydra.core',
        'omegaconf',
        'iopath',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ImageCropTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64',
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ImageCropTool',
)

app = BUNDLE(
    coll,
    name='ImageCropTool.app',
    icon=None,
    bundle_identifier='com.croptool.imagecroptool',
    version='1.0.0',
    info_plist={
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15.0',
        'CFBundleShortVersionString': '1.0.0',
    },
)