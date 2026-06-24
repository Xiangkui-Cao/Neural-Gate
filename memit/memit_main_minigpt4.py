import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minigpt4.conversation.conversation import CONV_VISION_LLama2
from rome.layer_stats import layer_stats
from util import nethook
# from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z_minigpt import compute_z, get_module_input_output_at_words, find_fact_lookup_idx, get_module
from .memit_hparams import MEMITHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

# print(COV_CACHE)

def apply_memit_to_model(
    model,
    tok: AutoTokenizer,
    image_embeddings,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)
    deltas = execute_memit(model, tok, image_embeddings, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model.model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float().to(w.device)

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_memit(
    model,
    tok: AutoTokenizer,
    image_embeddings,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    # requests = deepcopy(requests)
    # for i, request in enumerate(requests):
    #     if request["target_new"]["str"][0] != " ":
    #         # Space required for correct tokenization
    #         requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model.model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok, image_embeddings)
    # context_templates = [[""], ["", "", "", "", ""]]
    z_layer = hparams.layers[-1]
    z_list = []

    for request, image_embedding in zip(requests, image_embeddings):
        print(f"now request:{request}")
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                image_embedding,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)
    # layers = [hparams.rewrite_module_tmp.format(layer) for layer in hparams.layers]
    # covs = get_cov(model, tok, layers, COV_DATA_PATH, COV_PATH)
    # Insert
    init_cov(model)
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, image_embeddings, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        inputs = [model.get_conv([request["prompt"]], [""])[0].get_prompt() for request in requests]
        cur_zs_list = []
        for image_embedding, input, request in zip(image_embeddings, inputs, requests):
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                [image_embedding],
                z_layer,
                context_templates=[input],
                words=[request["subject"]],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1]
            cur_zs_list.append(cur_zs)
        cur_zs = torch.cat(cur_zs_list, dim=0).T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix
        force_recompute = False

        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        )

        # Compute update in double precision
        layer_ks, targets = (
            layer_ks.double().to(cov.device),
            targets.double().to(cov.device),
        )
        # cov = covs[hparams.rewrite_module_tmp.format(layer)].to(model.device)
        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
            layer_ks,
        )
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        print(f"resid shape: {resid.shape}")
        print(f"adj_k shape: {adj_k.shape}")
        print(f"weight shape: {weights[weight_name].shape}")
        print(f"upd_matrix shape: {upd_matrix.shape}")
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("resid norm", torch.linalg.norm(resid))
        print("adj_k norm", torch.linalg.norm(adj_k))
        print("cov norm", torch.linalg.norm(hparams.mom2_update_weight * cov.double()))
        print("layer_ks @ layer_ks.T norm", torch.linalg.norm(layer_ks @ layer_ks.T))
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable

        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.half().to(weights_copy[weight_name].device)
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()


    # save cov
    # if not os.path.exists("cov"):
    #     os.mkdir("cov")
    # torch.save(COV_CACHE, "cov/cov_cache.pth")
    # Saving Cache
    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas

def init_cov(model):
    global COV_CACHE
    if model.model_name == "minigpt4-llama2-7b" and os.path.exists("cov/cov_cache_minigpt.pth"):
        COV_CACHE = torch.load("cov/cov_cache_minigpt.pth")
    elif model.model_name == "llava1.5-7b" and os.path.exists("cov/cov_cache_llava.pth"):
        COV_CACHE = torch.load("cov/cov_cache_llava.pth")


def get_cov(
    model,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = "minigpt4" if model.model_name == "minigpt4-llama2-7b" else "llava1.5"
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            "wikitext-103-row-v1",
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
        if not os.path.exists("cov"):
            os.mkdir("cov")
        if model.model_name == "llava1.5-7b":
            torch.save(COV_CACHE, "cov/cov_cache_llava.pth")
        else:
            torch.save(COV_CACHE, "cov/cov_cache_minigpt.pth")

    return (
        torch.inverse(COV_CACHE[key].to(model.device)) if inv else COV_CACHE[key].to(model.device)
    )

# def get_cov(model, tok, layers, data_path, load_path="weights/minigpt_cov.pth"):
#     if os.path.exists(load_path):
#         print("loading COV...")
#         return torch.load(load_path)
#
#     print("Generating COV...")
#     def hook_fn(module, input, output):
#         nonlocal layer_input
#         nonlocal layer_idx
#         # print(input[0].shape)
#         layer_input[layer_idx] = input[0].detach().clone()
#         layer_idx = layer_idx + 1 if layer_idx < len(layers) - 1 else 0
#
#     module_out = []
#     layer_input = [None for _ in layers]
#     hooks = []
#     for i, layer in enumerate(layers):
#         module_out.append(get_module(model.model, layer))
#         hooks.append(module_out[i].register_forward_hook(hook_fn))
#
#     with open(data_path) as f:
#         cov_dataset = json.load(f)
#
#     cov_out = {}
#     layer_idx = 0
#     for i, case in enumerate(cov_dataset):
#         inp = case["answer"][:30]
#         conv = CONV_VISION_LLama2.copy()
#         conv.append_message(conv.roles[0], inp)
#         conv.append_message(conv.roles[1], "")
#         prompt = conv.get_prompt()
#         inpt_toks = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
#         input_embs = model.model.embed_tokens(inpt_toks).detach()
#         outputs = model.model.llama_model(
#             inputs_embeds=input_embs
#         )
#         for j, layer in enumerate(layers):
#             layer_input[j] = torch.mean(layer_input[j], dim=1)
#             layer_matrix = layer_input[j].T @ layer_input[j]
#             if layer not in cov_out.keys():
#                 cov_out[layer] = torch.zeros_like(layer_matrix).type_as(layer_matrix).to(model.device)
#             cov_out[layer] = cov_out[layer] + (layer_matrix - cov_out[layer]) / (i + 1)
#             # cov_out[layer] = cov_out[layer] + layer_matrix
#
#     if not os.path.exists(os.path.split(load_path)[0]):
#         os.makedirs(os.path.split(load_path)[0])
#     print("Saving COV...")
#     torch.save(cov_out, load_path)
#     for hook in hooks:
#         hook.remove()
#
#     return cov_out


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok, imgs):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"], ["The image shows that. {}", "Therefore, it is. {}", "Because I am just. {}",
                                            "You want to. {}"]]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

