import hashlib
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project_root = Path(SPECPATH).parent
model_path = (
    project_root
    / "models"
    / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
model_size = 491400032
model_sha256 = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"

if not model_path.is_file():
    raise SystemExit(
        "Required Qwen model is missing. Download the documented GGUF file "
        f"before packaging: {model_path}"
    )
if model_path.stat().st_size != model_size:
    raise SystemExit(
        f"The Qwen model has an unexpected size: {model_path}. "
        "Download the verified model file before packaging."
    )

digest = hashlib.sha256()
with model_path.open("rb") as model_file:
    for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != model_sha256:
    raise SystemExit(
        "The Qwen model checksum does not match the documented file. "
        "Do not package an unverified model."
    )

package_binaries = []
package_datas = [
    (str(model_path), "models"),
    (str(project_root / "config" / "bloom_levels.json"), "config"),
    (str(project_root / "config" / "bloom_levels.default.json"), "config"),
    (str(project_root / "packaging" / "CO-BL-Predictor.ico"), "."),
]
package_hiddenimports = []
for package_name in ("llama_cpp", "pymupdf", "sklearn"):
    binaries, datas, hiddenimports = collect_all(package_name)
    package_binaries.extend(binaries)
    package_datas.extend(datas)
    package_hiddenimports.extend(hiddenimports)

analysis = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=package_binaries,
    datas=package_datas,
    hiddenimports=package_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(analysis.pure)
executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="CO-BL-Predictor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "packaging" / "CO-BL-Predictor.ico"),
    version=str(project_root / "packaging" / "version_info.txt"),
    contents_directory="_internal",
)
COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="CO-BL-Predictor",
)
