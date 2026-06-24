import torch
# from rome.repr_tools import get_conv_embedding_with_image
# from rome import repr_tools
import numpy as np
from copy import deepcopy
import re

def get_ids(model, request, tok, add_special_tokens, return_tensors="pt"):
    assert isinstance(request, str)
    if model.model_name == "llava1.5-7b":
        ids_list = tok(request, add_special_tokens=add_special_tokens).input_ids
        return torch.tensor(ids_list, dtype=torch.long).unsqueeze(0).to("cuda")
    else:
        return tok(request, return_tensors=return_tensors, add_special_tokens=add_special_tokens).input_ids.to("cuda")

def get_conv_embedding_with_image_model(model, tok, image_embedding, inputs, outputs):
    if model.model_name == "llava1.5-7b":
        return model.get_conv_embedding_with_image(tok, image_embedding, inputs, outputs)
    else:
        return get_conv_embedding_with_image(model, tok, image_embedding, inputs, outputs)

def model_outputs(model, inputs_embeds, attention_mask):
    if model.model_name == "llava1.5-7b":
        if hasattr(model, "base_model"):
            input_device = next(model.base_model.model.model.layers[0].parameters()).device
            outputs = model.base_model.model.model(
                inputs_embeds=inputs_embeds.to(input_device),
                attention_mask=attention_mask.to(input_device),
                return_dict=True
            )
            # print("Output keys:", outputs.keys())
            hidden_state = outputs[0]

            class OUT:
                def __init__(self, outputs):
                    self.logits = outputs

            return OUT(model.base_model.model.lm_head(hidden_state))
        else:
            input_device = next(model.model.model.layers[0].parameters()).device
            outputs =  model.model.model(
                inputs_embeds=inputs_embeds.to(input_device),
                attention_mask=attention_mask.to(input_device),
                return_dict=True
            )
            # print("Output keys:", outputs.keys())
            hidden_state = outputs[0]
            class OUT:
                def __init__(self, outputs):
                    self.logits = outputs
            assert model.model.lm_head(hidden_state).shape[-1] == model.tokenizer.vocab_size, f"output shape: {model.model.lm_head(hidden_state).shape}"
            return OUT(model.model.lm_head(hidden_state))
    else:
        if hasattr(model, "llama_model") and hasattr(model.llama_model, "base_model"):
            input_device = next(model.llama_model.base_model.model.model.layers[0].parameters()).device
            return model.llama_model.base_model(
            inputs_embeds=inputs_embeds.to(input_device),
            attention_mask=attention_mask.to(input_device),
            )
        else:
            input_device = next(model.model.llama_model.model.layers[0].parameters()).device
            return model.model.llama_model(
                inputs_embeds=inputs_embeds.to(input_device),
                attention_mask=attention_mask.to(input_device)
            )

def find_fact_lookup_idx_model(
        model,
        prompt: str,
        subject: str,
        tok,
        fact_token_strategy: str,
        verbose=True,
        function=None
):
    if model.model_name != "llava1.5-7b":
        return find_fact_lookup_idx(model, prompt, subject, tok, fact_token_strategy, verbose)
    if model.model_name == "llava1.5-7b":
        return model.find_fact_lookup_idx(prompt, subject, tok, fact_token_strategy, verbose)
    raise Exception

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
        ret = get_words_idxs_in_templates(
            model=model,
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

    return ret + Image_toks + 1 if ret != -1 else -1

def get_words_idxs_in_templates(
    model, tok, context_templates: str, words: str, subtoken: str
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