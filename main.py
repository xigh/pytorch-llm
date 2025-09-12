import torch
import json
import time
import argparse

from model3 import Gemma3Model
from config import load_repo_id
from modelsize import model_memory_size
from tokenizer import load_tokenizer, apply_chat_template
from load import load_weights_into_gemma

# default_repo_id = f"google/gemma-3-270m-it"
default_repo_id = f"google/gemma-3-1B-it"
# default_repo_id = f"google/gemma-3-4B-it" # <----- NOT COMPATIBLE YET

default_prompt="Give me a short introduction to large language models."

# ------------------------------------------------------

def positive_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", type=str, default=default_repo_id, help="ID du modèle à utiliser")
parser.add_argument("--prompt", type=str, default=default_prompt, help="prompt")
parser.add_argument("--show-debug", action="store_true", help="show debugging logs")
parser.add_argument("--show-logits", type=positive_int, default=0, help="show logits")
parser.add_argument("--verbose", action="store_true", help="show timings")
parser.add_argument("--compile", action="store_true", help="compile model")
parser.add_argument("--force-cpu", action="store_true", help="force cpu inference")
args = parser.parse_args()

config, weights_dict, local_dir = load_repo_id(args.repo_id)

# ------------------------------------------------------

torch.manual_seed(123)
model = Gemma3Model(config)

# ------------------------------------------------------

load_weights_into_gemma(model, config, weights_dict)
del weights_dict

# ------------------------------------------------------

total_params = sum(p.numel() for p in model.parameters())

if args.show_debug:
    print(f"Total number of parameters: {total_params:,}")
# Account for weight tying
total_params_normalized = total_params - model.tok_emb.weight.numel()
if args.show_debug:
    print(f"\nTotal number of unique parameters: {total_params_normalized:,}")

if args.show_debug:
    print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
    print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")

# ------------------------------------------------------

if args.force_cpu:
    device = torch.device("cpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ------------------------------------------------------

if args.show_debug:
    print("Moving model to device")
model.to(device)

# ------------------------------------------------------

if args.show_debug:
    print("Loading tokenizer")
tokenizer = load_tokenizer(local_dir, args.repo_id)

# ------------------------------------------------------

prompt = apply_chat_template(args.prompt)

if args.show_debug:
    print("encoding tokens")
input_token_ids = tokenizer.encode(prompt)

# ------------------------------------------------------

# Test is encode/decode is working
# text = tokenizer.decode(input_token_ids)
# print(text)

# ------------------------------------------------------

# Optionally use torch.compile for an extra speed-up
if args.compile:
    if device.type == "mps":
        print("warning MPS device detected - compilation does not work")
    try:
        print("compiling model with 'max-autotune' ...")
        model = torch.compile(model, backend="inductor", mode="max-autotune")
        print("compilation finished")
    except Exception as e:
        print(f"compilation failed with 'max-autotune': {e}")
        try:
            print("compiling model with 'default' ...")
            model = torch.compile(model, backend="inductor", mode="default")
            print("compilation finished")
        except Exception as e2:
            print("compilation failed with 'default'")

# ------------------------------------------------------

def generate_text_basic_stream(model, token_ids, max_new_tokens, tokenizer, eos_token_id=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]
            if args.show_logits > 0:
                print("")
                topk = torch.topk(out, args.show_logits, dim=-1)
                values, indices = topk.values[0], topk.indices[0]
                print("Top logits:")
                for score, idx in zip(values.tolist(), indices.tolist()):
                    token = tokenizer.decode([idx])
                    quoted = json.dumps(token)
                    print(f"\t{idx:10} {quoted:25} {score:.3f}")

            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if (eos_token_id is not None
                   and torch.all(next_token == eos_token_id)):
               break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)

# ------------------------------------------------------

if args.show_debug:
    print("processing input")
input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

if args.compile:
    print("warming up")
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in generate_text_basic_stream(
            model=model,
            token_ids=input_token_ids_tensor,
            max_new_tokens=5,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.encode("<end_of_turn>")[-1],
        ):
            pass
        if device.type == "cuda":
            torch.cuda.synchronize()
    print("JIT ready")

if args.show_debug:
    print("generating response")

times = []
first_token_time = None
last_time = time.perf_counter()
start_time = last_time
n_tokens = 0

for i, token in enumerate(generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=500,
    tokenizer=tokenizer,
    eos_token_id=tokenizer.encode("<end_of_turn>")[-1]
)):
    # timings
    now = time.perf_counter()
    n_tokens += 1
    if i == 0:
        first_token_time = now - last_time
    else:
        times.append(now - last_time)
    last_time = now

    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )

if args.verbose:
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    print("\n")
    if first_token_time is not None:
        print(f"time to first token : {first_token_time:.4f} s")

    if times:
        print(f"min time between tokens : {min(times):.4f} s")
        print(f"max time between tokens : {max(times):.4f} s")
        print(f"avg time between tokens : {sum(times)/len(times):.4f} s")

    if n_tokens > 0 and len(times) > 0:
        generation_time = sum(times)
        generation_speed = len(times) / generation_time
        print(f"{generation_speed:.2f} tokens/s")
