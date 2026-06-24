import json
from io import BytesIO
import csv
import shutil
from copy import deepcopy
from itertools import islice
from time import time
from typing import Tuple, Union
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,7'

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import numpy as np
from PIL import Image
from transformers import StoppingCriteriaList, TextIteratorStreamer
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, CenterCrop
# from transformers import AutoModelForCausalLM, AutoTokenizer
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

from ft.ft_haparams import FTHyperParams
from ft.ft_main import apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dinm.dinm_hparams import DINMHyperParams
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
    ScienceQA,
    MLLMGUARD,
    privacy_ds_cat
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_memit_to_model
from memit.memit_tools import generate_fast
from rome import ROMEHyperParams, apply_rome_to_model
from dinm import apply_dinm_to_model
from ours.mymethod import apply_ours_to_model
from ours.mymethod_bp import apply_bp_to_model
from ours.mymethod_abs import apply_abs_to_model
from ours.mymethod_bp_cat import apply_bp_cat_to_model
from util import nethook
from util.globals import *

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    "Ours": (MEMITHyperParams, apply_ours_to_model),
    "DINM": (DINMHyperParams, apply_dinm_to_model),
    "Ours_bp": (DINMHyperParams, apply_bp_to_model),
    "Ours_bp_cat": (DINMHyperParams, apply_bp_cat_to_model),
    "Ours_abs": (DINMHyperParams, apply_abs_to_model)
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "privacy": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "science_qa": (ScienceQA, compute_rewrite_quality_counterfact),
    "mllmguard": (MLLMGUARD, compute_rewrite_quality_counterfact)
}


class StreamingInterSampleStats:
    def __init__(self, num_layers=17, dim=4096, device="cpu"):
        self.num_layers = num_layers
        self.dim = dim
        self.device = device

        self.count = 0
        self.mean = torch.zeros(num_layers, dim, device=device)  # 均值
        self.M2 = torch.zeros(num_layers, dim, device=device)  # 平方和偏差

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        assert x.shape == (self.num_layers, self.dim), f"输入维度错误：{x.shape} != ({self.num_layers}, {self.dim})"

        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_stats(self):
        if self.count < 2:
            variance = torch.zeros_like(self.mean)
            std = torch.zeros_like(self.mean)
        else:
            variance = self.M2 / (self.count - 1)
            std = torch.sqrt(variance)

        return {
            "mean": self.mean,
            "variance": variance,
            "std": std,
            "count": self.count
        }


class StreamingLayerStats:
    def __init__(self, num_layers=17, dim=4096, device="cpu"):
        self.tracker = StreamingInterSampleStats(
            num_layers=num_layers, dim=dim, device=device
        )

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        self.tracker.update(x)

    def get_stats(self):
        stats = self.tracker.get_stats()

        layer_mean = stats["mean"].mean(dim=1)  # [num_layers]
        layer_variance = stats["variance"].mean(dim=1)  # [num_layers]
        layer_std = stats["std"].mean(dim=1)  # [num_layers]

        return {
            "layer_mean": layer_mean,
            "layer_variance": layer_variance,
            "layer_std": layer_std,
            "total_count": stats["count"],
            "full_mean": stats["mean"],
            "full_variance": stats["variance"]
        }


class MiniGPT(nn.Module):
    def __init__(self, args):
        super(MiniGPT, self).__init__()

        # ========================================
        #             Model Initialization
        # ========================================
        self.model_name = "minigpt4-llama2-7b"
        conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                     'pretrain_llama2': CONV_VISION_LLama2}

        random_number = random.randint(1, 2000)
        random.seed(random_number)
        np.random.seed(random_number)
        torch.manual_seed(random_number)

        cudnn.benchmark = False
        cudnn.deterministic = True

        print('Initializing Chat')
        cfg = Config(args)

        self.device = 'cuda:{}'.format(args.gpu_id)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to(self.device)
        # self.model.vit_model = self.model.vit_model.to("cuda:0")

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        self.model = self.model.eval()

        CONV_VISION = conv_dict[model_config.model_type]

        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(self.device) for ids in stop_words_ids]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        if stopping_criteria is not None:
            self.stopping_criteria = stopping_criteria
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        print('Initialization Finished')

        self.conv = CONV_VISION.copy()
        self.conv.append_message(self.conv.roles[0], "<Img><ImageHere></Img>")

    def get_conv(self, inputs, outputs):
        output = []
        for inp, outp in zip(inputs, outputs):
            conv = self.conv.copy()
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], inp])
            conv.append_message(conv.roles[1], outp)
            output.append(conv)
        return output

    def image_transform(
            self,
            image_size,
            mean,
            std,
    ):
        normalize = Normalize(mean=mean, std=std)
        transforms = [
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
        ]

        def _convert_to_rgb(image):
            return image.convert('RGB')

        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)

    def get_images(self, paths):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        outputs = []
        for path in paths:
            image = Image.open(path)
            image = self.image_transform([224, 224], mean, std)(image).unsqueeze(0).to(self.device)
            outputs.append(image)
        return outputs

    def get_pil(self, pils):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        outputs = []
        for pil in pils:
            image = pil
            image = self.image_transform([224, 224], mean, std)(image).unsqueeze(0).to(self.device)
            outputs.append(image)
        return outputs

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        # print(prompt)
        prompt_segs = prompt.split('<ImageHere>')
        # print(prompt_segs)
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]

        inputs_tokens = []
        inputs_tokens.append(seg_tokens[0])
        # inputs_tokens.append( torch.from_numpy(np.ones((1,32))*(-200)).to(self.device) ) #for 224*224 num_Vtokens=32
        inputs_tokens.append(torch.from_numpy(np.ones((1, 64)) * (-200)).to(self.device))  # for 448*448 num_Vtokens=256
        inputs_tokens.append(seg_tokens[1])

        dtype = inputs_tokens[0].dtype
        inputs_tokens = torch.cat(inputs_tokens, dim=1).to(dtype)
        # print(inputs_tokens)
        # print(inputs_tokens.shape)
        seg_embs = [self.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        # print(seg_embs[0].shape)
        # print(f"{img_list[0].shape}")
        assert img_list[0].shape[1] == 64
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs, inputs_tokens

    def forward(self, embs, attns):
        max_new_tokens = 300
        min_length = 1
        max_length = 2000

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]
        attns = attns[:, begin_idx:]

        outputs = self.model.llama_model(inputs_embeds=embs, attention_mask=attns)
        return outputs.logits


def main(
        alg_name: str,
        model_name: Union[str, Tuple],
        hparams_fname: str,
        ds_name: str,
        dataset_size_limit: int,
        questions_per_img: int,
        continue_from_run: str,
        skip_generation_tests: bool,
        generation_test_interval: int,
        conserve_memory: bool,
        args,
        dir_name: str,
        num_edits: int = 1,
        use_cache: bool = False,
        privacy_scale=None,
        answer=None,
        layer_check=None,
        science_qa=False,
        mllmguard=False,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]
    if args.topk is not None and "Ours" in alg_name:
        topk = args.topk
    else:
        topk = None

    # Get run hyperparameters
    params_path = HPARAMS_DIR / alg_name / hparams_fname

    hparams = params_class.from_json(params_path)

    hparams.layers = [i for i in range(3, 20)]
    print(f"Executing {alg_name} with parameters {hparams}")

    model = MiniGPT(args)
    tok = model.model.llama_tokenizer

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    # snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    # vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    dataset_files = os.listdir(DATA_DIR)
    ds = ds_class(DATA_DIR, model=model, tok=tok, size=dataset_size_limit, files=dataset_files)
    if privacy_scale:
        for d in ds.data:
            d["requested_rewrite"]["subject_before"] = d["requested_rewrite"]["subject"]
            d["requested_rewrite"]["subject"] = privacy_scale
        assert ds.data[0]["requested_rewrite"]["subject"] == privacy_scale
    # Get cache templates

    print(f"num_edit: {num_edits}")
    edited_model = model
    if alg_name in ["DINM", "Ours_bp", "Ours_abs"]:
        num_edits = 1
    large_ds = [
        dict(
            case_id=d["case_id"],
            image=d["image"],
            requested_rewrite=dict(
                prompt=d["paraphrase_prompts"][i].replace(d["requested_rewrite"]["subject"], "{}", 1),
                target_new=d["requested_rewrite"]["target_new"],
                subject=d["requested_rewrite"]["subject"]
            ),
            neighborhood_prompts=[d["neighborhood_prompts"][i]],
            neighborhood_answers=[d["neighborhood_answers"][i]],
        )
        for d in ds for i in range(len(d["paraphrase_prompts"]))
    ]
    from model_func.model_func import get_ids, get_conv_embedding_with_image_model, model_outputs, \
        find_fact_lookup_idx_model
    cos_losses = dict()
    # save_dict = []
    cats = ["phone", "Receipts", "StudentID", "military", "document", "Passport"]
    lens = [30, 50, 20, 20, 50, 50]
    starts = [0, 30, 80, 100, 120, 170]
    assert len(large_ds) == sum(lens), len(large_ds)

    feature_dim = 4096
    for i_cat, (st, len_t) in enumerate(zip(starts, lens)):
        layer_stats_tracker = StreamingLayerStats(
            num_layers=17,
            dim=feature_dim,
            device="cpu"
        )

        for edit_idx, d in enumerate(large_ds[st:st + len_t]):
            # Compute weight changes + record weights that changed`
            # case_ids = [record["case_id"] for record in record_chunks]

            img_paths = [os.path.join(IMAGE_ROOT, d["image"])]
            # print(img_paths)
            # print(f"Image root: {IMAGE_ROOT}")
            imgs = model.get_images(img_paths)
            # encode_img outputs tuple()
            if model.model_name == "llava1.5-7b":
                img_embs = [model.model.encode_images(img) for img in imgs]
            else:
                img_embs = [model.model.encode_img(img)[0][0].unsqueeze(0) for img in imgs]
            print(f"num images: {len(imgs)}")
            request = {
                "case_id": d["case_id"],
                "neighborhood_prompts": d["neighborhood_prompts"],
                "neighborhood_answers": d["neighborhood_answers"],
                **d["requested_rewrite"]
            }

            target_ids = get_ids(model, request["target_new"]["str"], tok, return_tensors="pt",
                                 add_special_tokens=False).to("cuda")

            # Compile list of rewriting and KL x/y pairs
            rewriting_p = [request["prompt"]]
            rewriting_tar = [tok.decode(target_ids[0, :-1], skip_special_tokens=True)]
            # kl_prompts = ["{} is a"]
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
            input_tok, input_attns, input_embs = get_conv_embedding_with_image_model(model, tok, img_embs,
                                                                                     all_prompts_subject, rewriting_tar)

            lookup_idxs = []
            # Compute rewriting targets
            rewriting_targets = torch.tensor(-100, device="cuda").repeat(
                len(rewriting_p), *input_tok.shape[1:]
            )
            for i in range(len(rewriting_p)):
                ex_len = input_attns[i].sum()
                # lookup_idxs.append(ex_len - target_ids.shape[1])
                rewriting_targets[i, ex_len - target_ids.shape[1]: ex_len] = target_ids[0, :]

            kl_targets = torch.tensor(-100, device="cuda").repeat(
                len(kl_prompts), *input_tok.shape[1:]
            )
            for i in range(len(kl_prompts)):
                ex_len = input_attns[len(rewriting_p) + i].sum()
                # lookup_idxs.append(ex_len - kl_ids.shape[1])
                kl_targets[i, ex_len - kl_ids.shape[1]: ex_len] = kl_ids[0, :]

            lookup_idxs = [
                find_fact_lookup_idx_model(
                    model, model.get_conv([prompt], [""])[0].get_prompt(), request["subject"], tok, hparams.fact_token,
                    verbose=i == 0
                )
                for i, prompt in enumerate(all_prompts)
            ]

            layer_outputs = dict()

            def create_hook(name):
                def edit_output_ft(module, input, cur_out):
                    layer_outputs[name] = []
                    for i, idx in enumerate(lookup_idxs):
                        layer_outputs[name].append(cur_out[0][i, idx, :].float().detach().cpu())

                return edit_output_ft

            # nethook.set_requires_grad(False, model.model)
            hooks = []
            hparams.layers = range(3, 20)
            for layer in hparams.layers:
                module = get_module(model.model, hparams.layer_module_tmp.format(layer))
                hook_func = create_hook(f"layer_{layer}")
                hook = module.register_forward_hook(hook_func)
                hooks.append(hook)

            input_embs = input_embs.detach()
            input_attns = input_attns.detach()
            logits = model_outputs(model, inputs_embeds=input_embs,
                                   attention_mask=input_attns).logits.to(model.device)
            # features = []
            v1 = []
            v2 = []
            for k, v in layer_outputs.items():
                assert len(v) == 2
                v1.append(v[0])
                v2.append(v[1])
            features = [torch.stack(v1, dim=0), torch.stack(v2, dim=0)]

            for feature in features:
                layer_stats_tracker.update(feature)

            for hook in hooks:
                hook.remove()

        final_stats = layer_stats_tracker.get_stats()

        save_data = {
            "layer_mean": final_stats["layer_mean"].tolist(),
            "layer_variance": final_stats["layer_variance"].tolist(),
            "layer_std": final_stats["layer_std"].tolist(),
            "total_samples": final_stats["total_count"],
            "full_mean": final_stats["full_mean"].tolist(),
            "full_variance": final_stats["full_variance"].tolist()
        }

        with open(f"minigpt_sample_stats_{cats[i_cat]}.json", "w") as f:
            json.dump(save_data, f, indent=4)



def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i: i + n]


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME", "FT", "MEND", "Ours", "DINM", "Ours_bp", "Ours_abs", "Ours_bp_cat"],
        default="Ours_bp_cat",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
             "where a new run_id is generated on each run. "
             "If continuing from previous run, specify the run_id in --continue_from_run.",
        # required=True,
    )
    parser.add_argument(
        "--privacy_scale",
        choices=[None, "personal privacy", "trade secret", "state secret"],
        default=None
    )
    parser.add_argument(
        "--questions_per_img",
        default=5,
    )
    parser.add_argument(
        "--science_qa",
        default=False
    )
    parser.add_argument(
        "--mllmguard",
        default=False
    )
    parser.add_argument(
        "--topk",
        default=None,
        type=float
    )
    parser.add_argument(
        "--privacy_value",
        default=0,
        type=float
    )
    parser.add_argument(
        "--answer",
        # choices=["multi", "single"],
        # default="I cannot assist",
        default=None
    )
    parser.add_argument(
        "--model_name",
        # choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="minigpt4_llama2_7b",
        help="Model to edit.",
        # required=True,
    )
    parser.add_argument("--cfg-path", default='eval_configs/minigpt4_llama2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=1, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="minigpt4_llama2_7b.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        # required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=10,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
             "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
             "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.questions_per_img,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        False,
        args,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        privacy_scale=args.privacy_scale,
        answer=args.answer,
        layer_check=7,
        science_qa=args.science_qa,
        mllmguard=args.mllmguard,
    )