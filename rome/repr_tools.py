"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

from copy import deepcopy
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from model_func.model_func import find_fact_lookup_idx_model


def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    image_embeddings,
    context_templates: List[str], # conv with imagehere
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """

    # if model.model_name == "llava1.5-7b":
    #     Image_toks = 24 * 24
    #     context_wo_img = [context.replace("<image>", "") for context in context_templates]
    # else:
    #     Image_toks = 64
    #     context_wo_img = [context.replace("<ImageHere>", "") for context in context_templates]
    # idxs = [[x[0] + Image_toks] for x in get_words_idxs_in_templates(model, tok, context_wo_img, words, subtoken)]
    idxs = [find_fact_lookup_idx_model(model, context_template, word, tok, "subject_"+subtoken) for word, context_template in zip(words, context_templates)]
    return get_reprs_at_idxs(
        model,
        tok,
        image_embeddings,
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs,
        layer,
        module_template,
        track,
    )


def get_words_idxs_in_templates(
    model, tok: AutoTokenizer, context_templates: str, words: str, subtoken: str
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """
    # conv without imagehere
    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), f"We currently do not support multiple fill-ins for context"

    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    prefixes, suffixes = [
        tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    words = deepcopy(words)

    # Pre-process tokens
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0:
            assert prefix[-1] == " "
            prefix = prefix[:-1]

            prefixes[i] = prefix
            # words[i] = f"{words[i].strip()}"

    # Tokenize to determine lengths
    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)
    from model_func.model_func import get_ids
    batch_tok = [get_ids(model, s, tok, return_tensors="pt", add_special_tokens=i<n) for i,s in enumerate([*prefixes, *words, *suffixes])]
    prefixes_tok, words_tok, suffixes_tok = [
        batch_tok[i : i + n] for i in range(0, n * 3, n)
    ]
    prefixes_len, words_len, suffixes_len = [
        [el.shape[1] for el in tok_list]
        for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    ]

    # Compute indices of last tokens
    if subtoken == "last" or subtoken == "first_after_last":
        return [
            [
                prefixes_len[i]
                + words_len[i]
                - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
            ]
            # If suffix is empty, there is no "first token after the last".
            # So, just return the last token of the word.
            for i in range(n)
        ]
    elif subtoken == "first":
        return [[prefixes_len[i]] for i in range(n)]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")

def get_conv_embedding_with_image(model, tok, image_embedding, inputs, outputs):
    # assert len(image_embedding) == 1
    if len(inputs) == len(outputs)+1:
        outputs.append(outputs[0])
    if len(image_embedding) == 1 and len(inputs) == len(outputs):
        image_embedding = [image_embedding[0] for _ in inputs]
    else:
        assert len(image_embedding) == len(inputs) == len(outputs), f"images@{len(image_embedding)} | inputs@{len(inputs)} | outputs@{len(outputs)}"
    convs = model.get_conv(inputs, outputs)
    all_prompts = [conv.get_prompt() for conv in convs]
    all_prompts_pre, all_prompts_suff = [], []
    for prompt in all_prompts:
        assert "<ImageHere>" in prompt
        x, y = prompt.split('<ImageHere>')
        all_prompts_pre.append(x)
        all_prompts_suff.append(y)

    tok_ids_pre = [tok(seg, return_tensors="pt", add_special_tokens=True).input_ids[0, :].to(model.device)
                   # only add bos to the first seg
                   for i, seg in enumerate(all_prompts_pre)
                   ]
    tok_ids_suffix = [
        tok(seg, return_tensors="pt", add_special_tokens=False).input_ids[0, :].to(model.device)
        # only add bos to the first seg
        for i, seg in enumerate(all_prompts_suff)
        ]
    Image_toks = 64
    # img_toks = torch.from_numpy(np.ones((64)) * (-200)).to(model.device)
    input_toks = [torch.cat([toks_pre, torch.from_numpy(np.ones(64) * (-200)).to(model.device), toks_suff], dim=0).type_as(toks_pre) for
                  toks_pre, toks_suff in zip(tok_ids_pre, tok_ids_suffix)]

    # tok.padding_side = "right"
    # input_tok = tok(
    #     [prompt.format(request["subject"]) for prompt in all_prompts],
    #     return_tensors="pt",
    #     padding="longest",
    #     add_special_tokens=False
    # ).to("cuda")
    pre_lens = [input_.shape[0] for input_ in tok_ids_pre]
    input_tok = torch.nn.utils.rnn.pad_sequence(input_toks, padding_value=tok.pad_token_id, batch_first=True).type_as(input_toks[0]).to(model.device)
    # print(f"input_tok[0] decode: {tok.decode(torch.cat([tok_ids_pre[0], tok_ids_suffix[0]]))}")
    input_attns = torch.where(input_tok != tok.pad_token_id, torch.ones_like(input_tok), torch.zeros_like(input_tok))

    pre_embedding = [model.model.embed_tokens(pre_tok.unsqueeze(0)) for pre_tok in tok_ids_pre]
    suffix_embedding = [model.model.embed_tokens(input_tok[prompt_id:prompt_id + 1, pre_len + Image_toks:]) for
                        prompt_id, pre_len in enumerate(pre_lens)]
    input_embs = torch.cat([torch.cat([pre, img, suffix], dim=1) for pre, img, suffix in
                            zip(pre_embedding, image_embedding, suffix_embedding)], dim=0)
    return input_tok, input_attns, input_embs

def conv_embedding_with_image(model, tok, image_embedding, convs):
    from model_func.model_func import get_ids
    assert len(image_embedding) == 1
    # if len(image_embedding) == 1:
    image_embedding = [image_embedding[0] for _ in convs]
    all_prompts = convs
    all_prompts_pre, all_prompts_suff = [], []
    for prompt in all_prompts:
        if model.model_name == "llava1.5-7b":
            x, y = prompt.split('<image>')
        else:
            x, y = prompt.split('<ImageHere>')
        all_prompts_pre.append(x)
        all_prompts_suff.append(y)
    tok_ids_pre = [get_ids(model, seg, tok, return_tensors="pt", add_special_tokens=True)[0, :].to(model.device)
                   # only add bos to the first seg
                   for i, seg in enumerate(all_prompts_pre)
                   ]
    tok_ids_suffix = [
        get_ids(model, seg, tok, return_tensors="pt", add_special_tokens=False)[0, :].to(model.device)
        # only add bos to the first seg
        for i, seg in enumerate(all_prompts_suff)
        ]
    if model.model_name == "llava1.5-7b":
        Image_toks = 24 * 24
    else:
        Image_toks = 64
    # img_toks = torch.from_numpy(np.ones((64)) * (-200)).to(model.device)
    input_toks = [torch.cat([toks_pre, torch.from_numpy(np.ones(Image_toks) * (-200)).to(model.device), toks_suff], dim=0).type_as(toks_pre) for
                  toks_pre, toks_suff in zip(tok_ids_pre, tok_ids_suffix)]

    # tok.padding_side = "right"
    # input_tok = tok(
    #     [prompt.format(request["subject"]) for prompt in all_prompts],
    #     return_tensors="pt",
    #     padding="longest",
    #     add_special_tokens=False
    # ).to("cuda")
    pre_lens = [input_.shape[0] for input_ in tok_ids_pre]
    input_tok = torch.nn.utils.rnn.pad_sequence(input_toks, padding_value=tok.pad_token_id, batch_first=True).type_as(input_toks[0]).to(model.device)
    input_attns = torch.where(input_tok != tok.pad_token_id, torch.ones_like(input_tok), torch.zeros_like(input_tok))

    if model.model_name == "llava1.5-7b":
        pre_embedding = [model.model.get_model().embed_tokens(pre_tok).unsqueeze(0) for pre_tok in tok_ids_pre]
        suffix_embedding = [model.model.get_model().embed_tokens(input_tok[prompt_id:prompt_id + 1, pre_len + Image_toks:]) for
                            prompt_id, pre_len in enumerate(pre_lens)]
    else:
        pre_embedding = [model.model.embed_tokens(pre_tok.unsqueeze(0)) for pre_tok in tok_ids_pre]
        suffix_embedding = [model.model.embed_tokens(input_tok[prompt_id:prompt_id + 1, pre_len + Image_toks:]) for
                        prompt_id, pre_len in enumerate(pre_lens)]
    # print(f"pre shape: {pre_embedding[0].shape}")
    # print(f"suf shape: {suffix_embedding[0].shape}")
    # print(f"img shape: {image_embedding[0].shape}")
    # input()
    input_embs = torch.cat([torch.cat([pre, img, suffix], dim=1) for pre, img, suffix in
                            zip(pre_embedding, image_embedding, suffix_embedding)], dim=0)
    return input_tok, input_attns, input_embs

def get_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    image_embeddings,
    contexts: List[str], # conv with img
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    track: str = "in"
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """

    def _batch(n):
        for i in range(0, len(contexts), n):
            yield contexts[i : i + n], idxs[i : i + n]

    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}

    def _process(cur_repr, batch_idxs, key):
        nonlocal to_return
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        # print(f"cur_repr shape: {cur_repr.shape}")
        for i, idx_list in enumerate(batch_idxs):
            to_return[key].append(cur_repr[i, idx_list])

    for batch_contexts, batch_idxs in _batch(n=128):
        tok.padding_side = "right"
        _, input_attns, input_embs = conv_embedding_with_image(model, tok, image_embeddings, batch_contexts)
        input_attns = input_attns.detach()
        input_embs = input_embs.detach()
        with torch.no_grad():
            with nethook.Trace(
                module=model.model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                # model(**contexts_tok)
                from model_func.model_func import model_outputs
                outputs = model_outputs(
                    model,
                    inputs_embeds=input_embs,
                    attention_mask=input_attns
                )
                logits = outputs.logits

        if tin:
            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")

    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}
    # to_return = {k: v.unsqueeze(1) for k, v in to_return.items() if v.ndim == 1}
    # print(f"to_return in shape: {to_return['in'].shape}")
    if len(to_return) == 1:
        return to_return["in"] if tin else to_return["out"]
    else:
        return to_return["in"], to_return["out"]
