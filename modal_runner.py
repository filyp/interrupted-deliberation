# run this file in repo root
import json
import os
import subprocess
import sys

import modal
import yaml

repo = "https://github.com/filyp/seek-and-destroy.git"
branch = "main"

image = (
    modal.Image.debian_slim()
    .run_commands("apt-get install -y git")
    .pip_install_from_requirements("requirements.txt")
    .env({"PYTHONPATH": "/root/code/src:${PYTHONPATH:-}"})
)
app = modal.App("example-get-started", image=image)


# no timeout
@app.function(gpu="L4", cpu=(1, 1), timeout=24 * 3600)
def remote_func():
    # clone repo
    subprocess.run(["git", "clone", repo, "/root/code"], check=True)
    os.chdir("/root/code")
    subprocess.run(["git", "checkout", branch], check=True)
    
    # # set environment variables for authentication
    # os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    # os.environ["WANDB_API_KEY"] = wandb_key

    import torch as pt
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pt.set_default_device("cuda")

    from utils.git_and_reproducibility import get_storage
    from utils.loss_fns import neg_cross_entropy_loss
    from utils.git_and_reproducibility import is_repo_clean

    storage = get_storage(db_url)

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)


@app.local_entrypoint()
def main():
    remote_func.remote()
