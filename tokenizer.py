import os
from pathlib import Path
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

class GemmaTokenizer:
    def __init__(self, tokenizer_file_path: str):
        tok_file = Path(tokenizer_file_path)
        self._tok = Tokenizer.from_file(str(tok_file))
        # Attempt to identify EOS and padding tokens
        eos_token = "<end_of_turn>"
        self.pad_token_id = eos_token
        self.eos_token_id = eos_token

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

def apply_chat_template(user_text):
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"

def load_tokenizer(local_dir, repo_id):
    tokenizer_file_path = os.path.join(local_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_file_path):
        try:
            tokenizer_file_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir)
        except Exception as e:
            print(f"Warning: failed to download tokenizer.json: {e}")
            tokenizer_file_path = "tokenizer.json"

    return GemmaTokenizer(tokenizer_file_path=tokenizer_file_path)
