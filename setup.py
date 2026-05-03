"""
Run this ONCE to set up the virtual environment and install all dependencies.
Usage: python setup.py
"""
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent


def run(cmd: list[str], **kwargs):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"ERROR: command failed with code {result.returncode}")
        sys.exit(1)


def main():
    venv_dir = ROOT / ".venv"

    # 1. Create venv
    if not venv_dir.exists():
        print("Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(venv_dir)])
    else:
        print("Virtual environment already exists.")

    # 2. Resolve pip path
    if sys.platform == "win32":
        pip = str(venv_dir / "Scripts" / "pip.exe")
        python = str(venv_dir / "Scripts" / "python.exe")
    else:
        pip = str(venv_dir / "bin" / "pip")
        python = str(venv_dir / "bin" / "python")

    # 3. Upgrade pip
    run([python, "-m", "pip", "install", "--upgrade", "pip"])

    # 4. Install PyTorch with CUDA 12.4 (compatible with driver 595.79 / CUDA 13.2)
    print("\nInstalling PyTorch with CUDA 12.4 support...")
    run([
        pip, "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124"
    ])

    # 5. Verify CUDA is available
    print("\nVerifying CUDA availability...")
    result = subprocess.run(
        [python, "-c",
         "import torch; print(f'PyTorch {torch.__version__}'); "
         "print(f'CUDA available: {torch.cuda.is_available()}'); "
         "print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if "CUDA available: False" in result.stdout:
        print("WARNING: CUDA not detected. Training will fall back to CPU (slow).")

    # 6. Install remaining requirements
    print("\nInstalling project requirements...")
    run([pip, "install", "-r", str(ROOT / "requirements.txt")])

    # 7. Install PageIndex from source
    print("\nInstalling PageIndex (vectorless RAG)...")
    try:
        run([pip, "install", "git+https://github.com/VectifyAI/PageIndex.git"])
    except SystemExit:
        print("WARNING: PageIndex install failed (optional for Phase 1). Skipping.")

    # 8. Configure Kaggle token
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        token = os.getenv("KAGGLE_TOKEN", "")
        if token:
            kaggle_json.write_text(f'{{"token": "{token}"}}')
            if sys.platform != "win32":
                os.chmod(kaggle_json, 0o600)
            print(f"Kaggle token written to {kaggle_json}")
        else:
            print("WARNING: KAGGLE_TOKEN not set in .env. Dataset downloads will fail.")
            print("  Set KAGGLE_TOKEN in .env and re-run, or download datasets manually.")

    print("\n✓ Setup complete!")
    print(f"\nActivate the environment:")
    if sys.platform == "win32":
        print(f"  .venv\\Scripts\\activate")
    else:
        print(f"  source .venv/bin/activate")
    print("\nNext step: python scripts/download_datasets.py")


if __name__ == "__main__":
    main()
