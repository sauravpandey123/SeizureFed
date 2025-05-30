import torch
import numpy as np
import hashlib

def prg(seed, shape):
    # Use the seed to initialize PRG and return pseudorandom tensor
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(shape, generator=g)


def generate_shared_seed(client_i, client_j):
    # Deterministic string for seed generation
    seed_str = f"shared_seed_{min(client_i, client_j)}_{max(client_i, client_j)}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)