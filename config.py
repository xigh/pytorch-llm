import os
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download
import json

def load_repo_id(repo_id):
    local_dir = f"models/{repo_id}"

    config_file = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        local_dir=local_dir,
    )
    with open(config_file, "r") as f:
        config = json.load(f)

    archs = config["architectures"]
    if False:
        print("archs", archs)
    assert "Gemma3ForCausalLM" in archs

    config["qk_norm"] = True

    if False:
        for k in config:
            print(f" # {k:30} -> {config[k]} [unknown]")

    if "Gemma3ForCausalLM" in archs:
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        weights_dict = load_file(weights_file)
    else:
        repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
        index_path = os.path.join(repo_dir, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)

        weights_dict = {}
        for filename in set(index["weight_map"].values()):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)

    return config, weights_dict, local_dir
