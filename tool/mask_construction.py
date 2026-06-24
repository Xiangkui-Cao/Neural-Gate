import torch

def min_k_mask(x, k):
    flat_x = x.view(-1)
    _, indices = torch.topk(flat_x, k, largest=False)
    mask = torch.zeros_like(flat_x, dtype=torch.bool)
    mask[indices] = True
    return mask.view_as(x)

def max_k_mask(x, k):
    flat_x = x.view(-1)
    _, indices = torch.topk(flat_x, k)
    mask = torch.zeros_like(flat_x, dtype=torch.bool)
    mask[indices] = True
    return mask.view_as(x)
