# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:/My Developments/VHR Maths Engine/main.py'],
    pathex=[],
    binaries=[],
    datas=[('C:/My Developments/VHR Maths Engine/assets', 'assets')],
    hiddenimports=['PIL', 'PIL.Image', 'PIL.ImageTk', 'matplotlib', 'matplotlib.backends.backend_tkagg', 'matplotlib.backends.backend_agg', 'scipy.fft', 'scipy.optimize'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VHR_Maths_Engine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='C:/My Developments/VHR Maths Engine/version_info.txt',
    icon=['C:/My Developments/VHR Maths Engine/assets/icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VHR_Maths_Engine',
)
