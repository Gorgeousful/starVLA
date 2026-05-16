from pathlib import Path
import os
import subprocess
import sys
import modal

APP_NAME = "starvla"
GPU_TYPE = "RTX-PRO-6000" # RTX-PRO-6000 H100 B200 LS40S
GPU_COUNT = 1
VOLUME_NAME = "starvla"
REMOTE_REPO_DIR = "/data0/luokang/research/starvla/starVLA"

LOCAL_REPO_ROOT = Path("/data0/luokang/research/starvla/starVLA")
DOCKERFILE = LOCAL_REPO_ROOT / "modal" / "Dockerfile.cu128"
TRAIN_SCRIPT = "examples/LIBERO-custom/train_files/run_libero_train.sh"

app = modal.App(APP_NAME)
storage = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.from_dockerfile(DOCKERFILE)
    .env(
        {
            "WANDB_API_KEY": "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"
        }
    )
    .add_local_dir(
        LOCAL_REPO_ROOT,
        REMOTE_REPO_DIR,
        copy=False,
        ignore=[
            "examples/LIBERO-custom/train_files/run_libero_train_modal.py", # ignore this script to avoid potential conflicts
            "playground/**",
            ".git/**",
        ],
    )
)


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={f"{REMOTE_REPO_DIR}/playground": storage},
    timeout=24*60*60
)
def train_libero():
    os.chdir(REMOTE_REPO_DIR)
    try:
        subprocess.run(
            ["bash", f"{REMOTE_REPO_DIR}/{TRAIN_SCRIPT}"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )
    finally:
        storage.commit()


@app.local_entrypoint()
def main():
    train_libero.remote()
