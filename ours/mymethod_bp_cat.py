from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque

import torch
from nltk.misc.chomsky import subjects
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.autograd import gradcheck
from apex import amp

from util import nethook

from rome.repr_tools import get_conv_embedding_with_image
# from compute_weights import compute_weights
# from mymethod import get_context_templates
from ours.mymethod import get_context_templates
from ours.compute_weights_cat import compute_weights_cat


# from trainer import kl_loc_loss, masked_log_probs


def apply_bp_cat_to_model(
        model,
        tok: AutoTokenizer,
        ds_list,
        questions_per_img,
        image_root,
        hparams,
        answer=None,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        privacy_value=0,
        topk=None,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original 1model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    for ds in ds_list:

        deltas = compute_weights_cat(model, tok, ds, int(questions_per_img), image_root, answer, hparams, hparams.layers, privacy_value=privacy_value, topk=topk)

        with torch.no_grad():
            for w_name, upd_matrix in deltas.items():
                w = nethook.get_parameter(model.model, w_name)
                if return_orig_weights and w_name not in weights_copy:
                    weights_copy[w_name] = w.detach().clone()

                w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

        if not keep_original_weight:
            weights_copy = {}

    return model, weights_copy


def get_edit_labels(tok, labels):
    return labels.masked_fill(labels == tok.pad_token_id, -100)


