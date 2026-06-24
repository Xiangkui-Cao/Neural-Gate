import os
from time import time
import json

import numpy as np

from tool.mask_construction import min_k_mask
from rome.repr_tools import get_conv_embedding_with_image
from rome import repr_tools
from copy import deepcopy
from accelerate import Accelerator
# import deepspeed

import torch.nn.functional as F
import torch
from model_func.model_func import get_ids, model_outputs, get_conv_embedding_with_image_model, find_fact_lookup_idx_model

def get_mask_for_cat(ds, model, tok, privacy_value, image_root, hparams, topk=None):
    masks_cat = []
    masks_org = []

    for edit_idx, record in enumerate(ds):
        # Compute weight changes + record weights that changed`
        privacy_value_dict = dict(privacy_value=privacy_value)
        topk_dict = dict(topk=topk) if topk is not None else dict()

        img_paths = [os.path.join(image_root, record["image"])]
        # print(img_paths)
        # print(f"Image root: {IMAGE_ROOT}")
        imgs = model.get_images(img_paths)
        # encode_img outputs tuple()
        if model.model_name == "llava1.5-7b":
            img_embs = [model.model.encode_images(img) for img in imgs]
        else:
            img_embs = [model.model.encode_img(img)[0][0].unsqueeze(0) for img in imgs]
        print(f"num images: {len(imgs)}")
        # for img_emb in img_embs:
        #     print(f"max@{max(img_emb)} min@{min(img_emb)}")

        start = time()
        edited_model = model
        mask_t, mask_org = get_mask_for_sample(
            edited_model,
            tok,
            img_embs,
    {"case_id": record["case_id"],
            "neighborhood_prompts": record["neighborhood_prompts"],
            "neighborhood_answers": record["neighborhood_answers"],
            **record["requested_rewrite"]},
            hparams,
            **privacy_value_dict,
            **topk_dict
        )
        masks_cat.append(mask_t)
        if mask_org is not None:
            masks_org.append(mask_org)

        exec_time = time() - start

    # for masks in masks_cat:
    masks_cat = [[mask.int() for mask in masks] for masks in masks_cat]
    masks_cat_mean = deepcopy(masks_cat[0])
    for masks in masks_cat[1:]:
        for mask_idx in range(len(masks)):
            # print(f"max mask {mask_idx}: {masks[mask_idx].max().item()}")
            masks_cat_mean[mask_idx] = masks_cat_mean[mask_idx] + masks[mask_idx]
            # print(f"after add max mask {mask_idx}: {masks_cat_mean[mask_idx].max().item()}")

    masks_cat_mean = [mask.float()/len(masks_cat) for mask in masks_cat_mean]
    print(f"mask dim: {masks_cat_mean[0].shape[-1]}")
    print(f"len masks_cat: {len(masks_cat)}")
    print(f"masks_cat_mean max: {masks_cat_mean[0].max()}")
    masks_cat_mean = [mask > 0.3 for mask in masks_cat_mean]
    print(f"masks_cat_mean[0]: {masks_cat_mean[0]}")

    return masks_cat_mean


def get_mask_for_sample(
        model,
        tok,
        img_embs,
        request,
        hparams,
        privacy_value,
        topk=None
):
    target_ids = get_ids(model, request["target_new"]["str"], tok, return_tensors="pt", add_special_tokens=False).to("cuda")
    context_templates = [["{}"], ["The image shows that. {}", "Therefore, it is. {}", "Because I am just. {}",
                                        "I am an AI. {}", "You want to. {}"]]
    # context_templates = [["{}"]]

    # Compile list of rewriting and KL x/y pairs
    rewriting_p = [
        context.format(request["prompt"])
        for context_types in context_templates
        for context in context_types
    ]
    rewriting_tar = [
        tok.decode(target_ids[0, :-1], skip_special_tokens=True)
        for context_types in context_templates
        for context in context_types
    ]
    print(f"rewriting_tar: {rewriting_tar}")

    kl_idx = 0
    if "subject_before" in request.keys():
        kl_prompts = [request["neighborhood_prompts"][kl_idx].replace(request["subject_before"], "{}", 1)]
    else:
        kl_prompts = [request["neighborhood_prompts"][kl_idx].replace(request["subject"], "{}", 1)]
    all_prompts = rewriting_p + kl_prompts
    # conv = model.get_conv(all_prompts, )
    all_prompts_subject = [prompt.format(request["subject"]) for prompt in all_prompts]

    kl_target = " ".join((request["neighborhood_answers"][kl_idx].split(" "))[:5])
    print(f"kl_target: {kl_target}")
    kl_ids = get_ids(model, kl_target, tok, return_tensors="pt", add_special_tokens=False).to("cuda")
    rewriting_tar.append(tok.decode(kl_ids[0, :-1], skip_special_tokens=True))
    assert len(rewriting_tar) == len(all_prompts)

    input_tok, input_attns, input_embs = get_conv_embedding_with_image_model(model, tok, img_embs,
                                                                       all_prompts_subject, rewriting_tar)
    print(f"input_embs shape: {input_embs.shape}")
    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_p), *input_tok.shape[1:]
    )
    for i in range(len(rewriting_p)):
        ex_len = input_attns[i].sum()
        rewriting_targets[i, ex_len - target_ids.shape[1]: ex_len] = target_ids[0, :]

    kl_targets = torch.tensor(-100, device="cuda").repeat(
        len(kl_prompts), *input_tok.shape[1:]
    )
    for i in range(len(kl_prompts)):
        ex_len = input_attns[len(rewriting_p) + i].sum()
        kl_targets[i, ex_len - kl_ids.shape[1]: ex_len] = kl_ids[0, :]
    lookup_idxs = [
        find_fact_lookup_idx_model(
            model, model.get_conv([prompt], [""])[0].get_prompt(), request["subject"], tok, hparams.fact_token, verbose=i == 0
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    layers = hparams.layers
    if model.model_name == "llava1.5-7b":
        masks = [torch.full((model.model.model.config.hidden_size,), 2, requires_grad=True, device=model.model.device,
                            dtype=torch.float16) for layer in layers]
    else:
        masks = [torch.full((model.model.llama_model.config.hidden_size,), 2, requires_grad=True, device=model.model.device,
                            dtype=torch.float16) for layer in layers]
    # print("mask is [1, ..., 1]")
    # mask_bs = [torch.ones_like(mask, device=mask.device) for mask in masks]
    # return mask_bs, None
    target_init, kl_distr_init = None, None

    def create_hook(mask):
        def edit_output_fn(module, input, cur_out):
            nonlocal target_init

            # if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                cur_out[0][i, idx, :] *= F.tanh(mask.to(cur_out[0].device))
                # print(f"cur_out shape: {cur_out[0].shape}")
            return cur_out

        return edit_output_fn

    # Optimizer
    opt = torch.optim.Adam(masks, lr=5e-1, weight_decay=0, eps=1e-3)

    # nethook.set_requires_grad(False, model.model)
    hooks = []
    for mask, layer in zip(masks, layers):
        module = get_module(model.model, hparams.layer_module_tmp.format(layer))
        hook_func = create_hook(mask)
        hook = module.register_forward_hook(hook_func)
        hooks.append(hook)

    # Execute optimization
    model.model.train()
    for param in model.model.parameters():
        param.requires_grad = False
    input_embs = input_embs.detach()
    input_attns = input_attns.detach()
    if_change = False
    edit_num = 25
    if model.model_name == "llava1.5-7b":
        accelerator = Accelerator()
        (
            model,
            opt,
        ) = accelerator.prepare(
            model, opt
        )
    max_edit_num = 100
    for it in range(edit_num):
        outputs = model_outputs(
            model,
            inputs_embeds=input_embs,
            attention_mask=input_attns
        )
        logits = outputs.logits.to(model.device)
        # print(f"logits shape: {logits.shape}")

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits[:len(rewriting_p)], dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(model.device),
        ).squeeze(2)
        target_mask = (rewriting_targets != -100).to(model.device)

        # Aggregate total losses
        nll_loss_each = -(loss * target_mask).sum(1) / target_ids.size(1)
        nll_loss = nll_loss_each.mean()
        post_logits = torch.log_softmax(logits[len(rewriting_p):, :, :], dim=2)
        print(f"post_logits: {post_logits.shape}")
        loss_kl = torch.gather(
            post_logits,
            2,
            torch.where(kl_targets != -100, kl_targets, 0).unsqueeze(2).to(model.device),
        ).squeeze(2)
        mask_kl = (kl_targets != -100).float().to(model.device)
        l_loc_instruction = -(loss_kl * mask_kl).sum(1)/kl_ids.size(1)
        mask_l1_loss = 0
        for mask in masks:
            mask_l1_loss += torch.abs(mask.to(model.device) - 2).mean()
        mask_l1_loss /= len(masks)
        loss = nll_loss + hparams.kl_factor * l_loc_instruction + 0.001 * mask_l1_loss

        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + "
            f"{np.round(l_loc_instruction.item(), 3)} + {np.round(0.001 * mask_l1_loss.item(), 3)}"
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss.item() < 5e-2:
            if_change = True
            break

        # Backpropagate
        opt.zero_grad()
        if model.model_name == "llava1.5-7b":
            accelerator.backward(loss)
        else:
            loss.backward()
        opt.step()
    for hook in hooks:
        hook.remove()
    print(
        f"<0@{torch.sum(F.tanh(masks[0]) < 0) / masks[0].shape[-1]} | <0.3@{torch.sum(F.tanh(masks[0]) < 0.3) / masks[0].shape[-1]}")

    print("Finetuning model based on mask")

    if topk is not None:
        mask_bs = [min_k_mask(mask, int(topk * mask.flatten().shape[-1])) for mask in masks]
        print(f"topk mask num: {mask_bs[0].int().sum().item()}")
    else:
        mask_bs = [F.tanh(mask.flatten()) < privacy_value for mask in masks]
        print(f"< {privacy_value} mask num: {mask_bs[0].int().sum().item()}")

    return mask_bs, [F.tanh(mask.half().detach().flatten()) for mask in masks]


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)

def find_fact_lookup_idx(
    model,
    prompt: str, # conv with imagehere
    subject: str,
    tok,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """
    Image_toks = 64
    prompt = prompt.replace("<ImageHere>", "")
    # print(fact_token_strategy)
    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(get_ids(model, sentence, tok, return_tensors="pt",
                                              add_special_tokens=True)[0, ret]),
        )

    return ret + Image_toks if ret != -1 else -1