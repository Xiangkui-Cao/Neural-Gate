from typing import Dict, List, Tuple
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from urllib3 import request

from rome import repr_tools
from util import nethook
from tool.mask_construction import min_k_mask

from .our_hparams import OurParams

from ours.get_mask_for_cat_level import get_mask_for_cat

from torch.cuda.amp import autocast, GradScaler
from model_func.model_func import get_ids, model_outputs, get_conv_embedding_with_image_model, find_fact_lookup_idx_model


def compute_weights_cat(
    model,
    tok: AutoTokenizer,
    ds,
    questions_per_img,
    image_root,
    answer,
    hparams: OurParams,
    layers: List[int],
    privacy_value=0,
    topk=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
    # assert privacy_value == -0.7
    # Tokenize target into list of int token IDs
    mask_bs = get_mask_for_cat(ds, model, tok, privacy_value, image_root, hparams, topk)
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model.model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    large_ds = [dict(
        case_id=d["case_id"],
        image=d["image"],
        requested_rewrite=dict(
            prompt=d["paraphrase_prompts"][i].replace(d["requested_rewrite"]["subject"], "{}", 1),
            target_new=d["requested_rewrite"]["target_new"],
            subject=d["requested_rewrite"]["subject"]
        ),
        neighborhood_prompts=[d["neighborhood_prompts"][i]],
        neighborhood_answers=[d["neighborhood_answers"][i]],
    ) for d in ds for i in range(questions_per_img)]

    # for d in ds:
    for d in large_ds:
        if answer:
            d["requested_rewrite"]["target_new"]["str"] = answer
        request = {
            "case_id": d["case_id"],
            "neighborhood_prompts": d["neighborhood_prompts"],
            "neighborhood_answers": d["neighborhood_answers"],
            **d["requested_rewrite"]
        }

        img_paths = [os.path.join(image_root, d["image"])]
        # print(img_paths)
        # print(f"Image root: {IMAGE_ROOT}")

        imgs = model.get_images(img_paths)
        # encode_img outputs tuple()
        if model.model_name == "llava1.5-7b":
            img_embs = [model.model.encode_images(img) for img in imgs]
        else:
            img_embs = [model.model.encode_img(img)[0][0].unsqueeze(0) for img in imgs]
        target_ids = get_ids(model, request["target_new"]["str"], tok, return_tensors="pt", add_special_tokens=False).to("cuda")

        rewriting_p  = [request["prompt"]]
        rewriting_tar = [tok.decode(target_ids[0, :-1], skip_special_tokens=True)]
        if "subject_before" in request.keys():
            kl_prompts = [request["neighborhood_prompts"][0].replace(request["subject_before"], "{}", 1)]
        else:
            kl_prompts = [request["neighborhood_prompts"][0].replace(request["subject"], "{}", 1)]
        all_prompts = rewriting_p + kl_prompts
        # conv = model.get_conv(all_prompts, )
        all_prompts_subject = [prompt.format(request["subject"]) for prompt in all_prompts]

        # kl_target = generate_fast(model, tok, [image_embedding], all_prompts_subject[-1:], max_out_len=5)[0]
        kl_target = " ".join((request["neighborhood_answers"][0].split(" "))[:5])
        kl_ids = get_ids(model, kl_target, tok, return_tensors="pt", add_special_tokens=False).to("cuda")
        rewriting_tar.append(tok.decode(kl_ids[0, :-1], skip_special_tokens=True))
        input_tok, input_attns, input_embs = get_conv_embedding_with_image_model(model, tok, img_embs, all_prompts_subject, rewriting_tar)

        # Compute rewriting targets
        rewriting_targets = torch.tensor(-100, device="cuda").repeat(
            len(rewriting_p), *input_tok.shape[1:]
        )
        for i in range(len(rewriting_p)):
            ex_len = input_attns[i].sum()
            rewriting_targets[i, ex_len - target_ids.shape[1] : ex_len] = target_ids[0, :]

        kl_targets = torch.tensor(-100, device="cuda").repeat(
            len(kl_prompts), *input_tok.shape[1:]
        )
        for i in range(len(kl_prompts)):
            ex_len = input_attns[len(rewriting_p) + i].sum()
            kl_targets[i, ex_len - kl_ids.shape[1]: ex_len] = kl_ids[0, :]

        # Compute indices of the tokens where the fact is looked up
        lookup_idxs = [
            find_fact_lookup_idx_model(
                model, model.get_conv([prompt], [""])[0].get_prompt(), request["subject"], tok, hparams.fact_token, verbose=i==0
            )
            for i, prompt in enumerate(all_prompts)
        ]
        # print(f"lookup_idxs: {lookup_idxs}")
        print(
            f"lookup idxs in compute weights token decode: {tok.decode(input_tok[0, lookup_idxs[0]])}")
        # Finalize rewrite and loss layers
        loss_layer = layers
        print(f"Rewrite layer is {layers}")
        print(f"Tying optimization objective to {loss_layer}")

        def create_hook(mask_b):
            def edit_output_ft(module, input, cur_out):
                for i, idx in enumerate(lookup_idxs):
                    # mask_b = torch.ones_like(cur_out[0][i, idx, :], dtype=torch.bool)
                    cur_out[0][i, idx, :] = torch.where(mask_b.to(cur_out[0].device), cur_out[0][i, idx, :], cur_out[0][i, idx, :].detach())
                return cur_out
            return edit_output_ft

        # nethook.set_requires_grad(False, model.model)
        hooks = []
        for mask, layer in zip(mask_bs, layers):
            module = get_module(model.model, hparams.layer_module_tmp.format(layer))
            hook_func = create_hook(mask)
            hook = module.register_forward_hook(hook_func)
            hooks.append(hook)


        # Save old weights for future restoration

        print(f"Weights to be updated: {list(weights.keys())}")

        opt = torch.optim.Adam(
            [v for _, v in weights.items()],
            lr=hparams.lr,
            weight_decay=hparams.weight_decay,
            # betas=(0.9, 0.999),
            eps=1e-3
        )

        for name, w in model.model.named_parameters():
            w.requires_grad = name in weights

        kl_nums = len(kl_prompts)

        edit_nums = len(rewriting_p)
        model.model.train()
        input_embs = input_embs.detach()
        input_attns = input_attns.detach()


        scaler = GradScaler()
        for it in range(hparams.num_steps):
            print(20 * "=")
            print(f"Epoch: {it}")
            print(20 * "=")

            with autocast(enabled=True):
                logits = model_outputs(model, inputs_embeds=input_embs,
                                                 attention_mask=input_attns).logits.to(model.device)  # torch.Size([1, 321, 32000])
                log_probs = torch.log_softmax(logits[:edit_nums], dim=2)
                # print(f"log_probs: {log_probs}")
                loss_t = torch.gather(
                    log_probs,
                    2,
                    torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(model.device),
                ).squeeze(2)
                mask_nll = (rewriting_targets != -100).float().to(model.device)
                # Aggregate total losses
                nll_loss_each = -((loss_t * mask_nll) / target_ids.size(1)).sum(1)
                nll_loss = nll_loss_each.mean()
                post_logits = torch.log_softmax(logits[edit_nums:], dim=2)
                # print(f"post_logits: {post_logits.shape}")
                # print(f"input attns: {input_attns}")
                loss_kl = torch.gather(
                    post_logits,
                    2,
                    torch.where(kl_targets != -100, kl_targets, 0).unsqueeze(2).to(model.device),
                ).squeeze(2)
                mask_kl = (kl_targets != -100).float().to(model.device)
                l_loc_instruction = -(loss_kl * mask_kl).sum(1)/kl_ids.size(1)
                # assert hparams.kl_factor == 1
                loss = nll_loss + hparams.kl_factor * l_loc_instruction
                # loss =  l_edit
                print(f"Batch loss {loss.item()}, loss_edit*0.1:{nll_loss}, loss_loc_instruction:{l_loc_instruction}")

            if loss.item() >= 1e-3:
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                break


        for hook in hooks:
            hook.remove()
    # Restore state of original model
    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    image_embedding,
    layer: int,
    context_templates: List[str], # conv with imagehere
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """
    context_templates = [context.replace("<ImageHere>", "") for context in context_templates]
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, image_embedding=image_embedding, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", image_embedding=image_embedding, **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    model,
    prompt: str, # conv with imagehere
    subject: str,
    tok: AutoTokenizer,
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
                                              add_special_tokens=True).input_ids[0, ret]),
        )

    return ret + Image_toks if ret != -1 else -1

def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)