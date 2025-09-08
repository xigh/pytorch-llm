import torch
import torch.nn as nn
import os

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dtype = torch.bfloat16
        if cfg["torch_dtype"] != "bfloat16":
            print(f"unexpected dtype {dtype}")
            os._exit(-1);

        self.fc1 = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], dtype=dtype, bias=False)
        self.fc2 = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], dtype=dtype, bias=False)
        self.fc3 = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"], dtype=dtype, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)
