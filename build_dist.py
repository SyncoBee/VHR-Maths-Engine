"""
VHR Maths Engine Pro - Automated Executable Packaging Suite
Created & Published by Vihanga Himantha

Automates compiling the Python source codebase into a standalone, native Windows executable (.exe)
with embedded icons, metadata, and dependencies.
"""
import os
import sys
import shutil
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
ICON_PATH = os.path.join(ASSETS_DIR, "icon.ico")
VERSION_FILE = os.path.join(ROOT_DIR, "version_info.txt")
ENTRY_POINT = os.path.join(ROOT_DIR, "main.py")
APP_NAME = "VHR_Maths_Engine"


def build_app(onefile: bool = False):
    print("=" * 70)
    print("  VHR Maths Engine Pro - Windows Executable Builder")
    print("  Created & Published by Vihanga Himantha")
    print("=" * 70)

    # 1. Verify icon exists
    if not os.path.exists(ICON_PATH):
        print(f"[!] Warning: Icon not found at {ICON_PATH}, generating default icon...")
        import scratch.gen_icon  # generate if missing

    # 2. Build PyInstaller command
    pyinstaller_exe = sys.executable.replace("python.exe", "pyinstaller.exe")
    if not os.path.exists(pyinstaller_exe):
        pyinstaller_exe = "pyinstaller"

    cmd = [
        pyinstaller_exe,
        "--noconfirm",
        "--clean",
        "--windowed",                 # No console window
        f"--name={APP_NAME}",
        f"--icon={ICON_PATH}",
        f"--version-file={VERSION_FILE}",
        f"--add-data={ASSETS_DIR};assets",
        # Explicit hidden imports for tkinter/matplotlib/scipy
        "--hidden-import=PIL",
        "--hidden-import=PIL.Image",
        "--hidden-import=PIL.ImageTk",
        "--hidden-import=matplotlib",
        "--hidden-import=matplotlib.backends.backend_tkagg",
        "--hidden-import=matplotlib.backends.backend_agg",
        "--hidden-import=scipy.fft",
        "--hidden-import=scipy.optimize",
    ]

    if onefile:
        print("[*] Target mode: Single Portable Executable (--onefile)")
        cmd.append("--onefile")
    else:
        print("[*] Target mode: Directory Distribution (--onedir - fast startup)")
        cmd.append("--onedir")

    cmd.append(ENTRY_POINT)

    print(f"[*] Executing PyInstaller command...")
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    if result.returncode != 0:
        print(f"[X] PyInstaller failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    dist_dir = os.path.join(ROOT_DIR, "dist")
    if onefile:
        exe_path = os.path.join(dist_dir, f"{APP_NAME}.exe")
        print(f"\n[+] SUCCESS! Portable executable built at:\n    {exe_path}")
    else:
        app_folder = os.path.join(dist_dir, APP_NAME)
        exe_path = os.path.join(app_folder, f"{APP_NAME}.exe")
        print(f"\n[+] SUCCESS! Application package built at:\n    {app_folder}")
        print(f"    Main Executable: {exe_path}")

    return exe_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Package VHR Maths Engine into Windows executable.")
    parser.add_argument("--onefile", action="store_true", help="Build single standalone .exe file.")
    args = parser.parse_args()
    build_app(onefile=args.onefile)
