import sys
import torch
import json

from model3 import Gemma3Model
from config import load_repo_id
from modelsize import model_memory_size
from tokenizer import load_tokenizer, apply_chat_template
from modelload import load_weights_into_gemma

SHOW_DEBUG=False
SHOW_LOGITS=0

# ------------------------------------------------------

model_name = "270m"
instruct_model = True

print(f"Loading model {model_name}{"-it" if instruct_model else ""} from HF")
config, weights_dict, repo_id, local_dir = load_repo_id(model_name, instruct_model)

# ------------------------------------------------------

torch.manual_seed(123)
model = Gemma3Model(config)

# ------------------------------------------------------

load_weights_into_gemma(model, config, weights_dict)
del weights_dict

# ------------------------------------------------------

total_params = sum(p.numel() for p in model.parameters())

if SHOW_DEBUG:
    print(f"Total number of parameters: {total_params:,}")
# Account for weight tying
total_params_normalized = total_params - model.tok_emb.weight.numel()
if SHOW_DEBUG:
    print(f"\nTotal number of unique parameters: {total_params_normalized:,}")

if SHOW_DEBUG:
    print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
    print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")

# ------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ------------------------------------------------------

if SHOW_DEBUG:
    print("Moving model to device")
model.to(device)

# ------------------------------------------------------

if SHOW_DEBUG:
    print("Loading tokenizer")
tokenizer = load_tokenizer(local_dir, repo_id)

# ------------------------------------------------------

prompt = "Give me a short introduction to large language models."
if len(sys.argv) > 1:
    prompt = sys.argv[1]
prompt = apply_chat_template(prompt)

if SHOW_DEBUG:
    print("encoding tokens")
input_token_ids = tokenizer.encode(prompt)

# ------------------------------------------------------

# Test is encode/decode is working
# text = tokenizer.decode(input_token_ids)
# print(text)

# ------------------------------------------------------

# Optionally use torch.compile for an extra speed-up
# model = torch.compile(model)
#

# ------------------------------------------------------

def generate_text_basic_stream(model, token_ids, max_new_tokens, tokenizer, eos_token_id=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]
            if SHOW_LOGITS > 0:
                print("")
                topk = torch.topk(out, SHOW_LOGITS, dim=-1)
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

if SHOW_DEBUG:
    print("processing input")
input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

if SHOW_DEBUG:
    print("generating response")
for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=500,
    tokenizer=tokenizer,
    eos_token_id=tokenizer.encode("<end_of_turn>")[-1]
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
