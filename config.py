import os
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download
import json

def load_repo_id(model_name, instruct_model):
    if instruct_model:
        repo_id = f"google/gemma-3-{model_name}-it"
    else:
        repo_id = f"google/gemma-3-{model_name}"

    local_dir = Path(repo_id).parts[-1]

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

    if model_name == "270m":
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

    return config, weights_dict, repo_id, local_dir
