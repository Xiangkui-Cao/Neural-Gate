from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook
from tool.mask_construction import min_k_mask

from .our_hparams import OurParams

import torch.nn.functional as F

from model_func.model_func import get_ids, find_fact_lookup_idx_model, get_conv_embedding_with_image_model,model_outputs

from memit.memit_tools import generate_fast

from minigpt4.conversation.conversation import CONV_VISION_Vicuna0

def compute_mask(
    model,
    tok: AutoTokenizer,
    image_embedding,
    request: Dict,
    hparams: OurParams,
    layer: int,
    context_templates: List[str],
    privacy_value=0,
    topk=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
    # Tokenize target into list of int token IDs
    target_ids = get_ids(model, request["target_new"]["str"], tok, return_tensors="pt", add_special_tokens=False).to("cuda")

    # Compile list of rewriting and KL x/y pairs
    rewriting_p  = [
        context.format(request["prompt"])
        for context_types in context_templates
        for context in context_types
    ]
    rewriting_tar = [
        tok.decode(target_ids[0, :-1], skip_special_tokens=True)
        for context_types in context_templates
        for context in context_types
    ]
    # kl_prompts = ["{} is a"]
    if "subject_before" in request.keys():
        kl_prompts = [request["neighborhood_prompts"][0].replace(request["subject_before"], "{}", 1)]
    else:
        kl_prompts = [request["neighborhood_prompts"][0].replace(request["subject"], "{}", 1)]
    all_prompts = rewriting_p + kl_prompts
    # conv = model.get_conv(all_prompts, )
    all_prompts_subject = [prompt.format(request["subject"]) for prompt in all_prompts]
    kl_target = " ".join((request["neighborhood_answers"][0].split(" "))[:5])
    kl_ids = get_ids(model, kl_target, tok, return_tensors="pt", add_special_tokens=False).to("cuda")
    rewriting_tar.append(tok.decode(kl_ids[0, :-1], skip_special_tokens=True))
    assert len(rewriting_tar) == len(all_prompts)

    input_tok, input_attns, input_embs = get_conv_embedding_with_image_model(model, tok, [image_embedding], all_prompts_subject, rewriting_tar)

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
        ex_len = input_attns[len(rewriting_p)+i].sum()
        kl_targets[i, ex_len - kl_ids.shape[1]: ex_len] = kl_ids[0, :]
    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx_model(
            model, model.get_conv([prompt], [""])[0].get_prompt(), request["subject"], tok, hparams.fact_token, verbose=i==0
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if model.model_name == "llava1.5-7b":
        mask = torch.full((model.model.model.config.hidden_size,), 2, requires_grad=True, device=model.device,
                          dtype=torch.float16)
    else:
        mask = torch.full((model.model.llama_model.config.hidden_size,), 2, requires_grad=True, device=model.device, dtype=torch.float16)
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
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
            cur_out[0][i, idx, :] *= F.tanh(mask)
            # print(f"cur_out shape: {cur_out[0].shape}")

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([mask], lr=hparams.v_lr, eps=1e-3)


    # nethook.set_requires_grad(False, model.model)
    module = get_module(model.model, hparams.layer_module_tmp.format(layer))
    hook = module.register_forward_hook(edit_output_fn)

    # Execute optimization
    model.model.train()
    input_embs = input_embs.detach()
    input_attns = input_attns.detach()
    for it in range(hparams.v_num_grad_steps):
        outputs = model_outputs(
            model,
            inputs_embeds=input_embs,
            attention_mask=input_attns
        )
        logits = outputs.logits

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits[:len(rewriting_p)], dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(model.device),
        ).squeeze(2)
        target_mask = (rewriting_targets != -100).float().to(model.device)

        # Aggregate total losses
        nll_loss_each = -(loss * target_mask).sum(1) / target_ids.size(1)
        nll_loss = nll_loss_each.mean()

        post_logits = torch.log_softmax(logits[len(rewriting_p):, :, :], dim=2)
        print(f"post_logits: {post_logits.shape}")
        print(f"input attns: {input_attns}")
        loss_kl = torch.gather(
            post_logits,
            2,
            torch.where(kl_targets != -100, kl_targets, 0).unsqueeze(2).to(model.device),
        ).squeeze(2)
        mask_kl = (kl_targets != -100).float().to(model.device)
        l_loc_instruction = -(loss_kl * mask_kl).sum(1)/kl_ids.size(1)
        # print(f"kl_log_probs: {kl_log_probs}")
        # assert hparams.kl_factor == 1
        # kl_loss = torch.nn.functional.kl_div(
        #     kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        # )
        loss = nll_loss + l_loc_instruction + 0.001 * torch.abs(mask-2).mean()
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(l_loc_instruction.item(), 3)}"
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )

        if loss < 5e-2:
            break

        # Backpropagate
        opt.zero_grad()
        loss.backward()
        opt.step()

    hook.remove()
    if topk is None:
        target = torch.where(F.tanh(mask) < privacy_value, target_init * F.tanh(mask), target_init)
    else:
        mask_b = min_k_mask(mask, int(topk*mask.flatten().shape[-1]))
        target = torch.where(mask_b, target_init * F.tanh(mask), target_init)
    # target = torch.where(mask < privacy_value, target_init * 0, target_init)
    print(
        f"Init norm {target_init.norm()} | Edited norm {target.norm()} | min mask {torch.min(F.tanh(mask))} | max mask {torch.max(F.tanh(mask))}"
    )
    print(f"<0.3@{torch.sum(F.tanh(mask) < 0.3)/mask.shape[-1]} | <0@{torch.sum(F.tanh(mask) < 0)/mask.shape[-1]}")

    return target


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
                                              add_special_tokens=True)[0, ret]),
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