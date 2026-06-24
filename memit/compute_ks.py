from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compute_z_minigpt import get_module_input_output_at_words
from .memit_hparams import MEMITHyperParams


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    image_embeddings,
    requests: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
):
    inputs = [model.get_conv([request["prompt"]], [""])[0].get_prompt() for request in requests]
    ks = []
    for image_embedding, input, request in zip(image_embeddings, inputs, requests):
        layer_ks = get_module_input_output_at_words(
            model,
            tok,
            [image_embedding],
            layer,
            context_templates=[
                input
                for context_type in context_templates
                for context in context_type
            ],
            words=[
                request["subject"]
                for context_type in context_templates
                for _ in context_type
            ],
            module_template=hparams.rewrite_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[0]
        ks.append(layer_ks)
    layer_ks = torch.cat(ks, dim=0)
    print(f"layer_ks shape: {layer_ks.shape}")
    # context templates = [[""], ["x", "y", "z", "m", "n"]]
    context_type_lens = [0] + [len(context_type) for context_type in context_templates]     # [0, 1, 5]
    context_len = sum(context_type_lens)    # 6
    context_type_csum = np.cumsum(context_type_lens).tolist()   # [0, 1, 6]

    ans = []
    for i in range(0, layer_ks.size(0), context_len):
        tmp = []
        for j in range(len(context_type_csum) - 1):
            start, end = context_type_csum[j], context_type_csum[j + 1]
            tmp.append(layer_ks[i + start : i + end].mean(0))
        ans.append(torch.stack(tmp, 0).mean(0))
    print(f"ans shape: {ans[0].shape}")
    return torch.stack(ans, dim=0)
