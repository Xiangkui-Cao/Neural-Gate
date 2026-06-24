import json
import csv
import shutil
import tokenize
from copy import deepcopy
from itertools import islice
from time import time
from typing import Tuple, Union
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import attention



import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import numpy as np
from PIL import Image
from transformers import StoppingCriteriaList, TextIteratorStreamer
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, CenterCrop
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from minigpt4.common.config import Config
# from minigpt4.common.dist_utils import get_rank
# from minigpt4.common.registry import registry
# from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

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
    POPE,
    MME_Dataset,
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
from alphaedit.AlphaEdit_main import apply_AlphaEdit_to_model
from alphaedit.AlphaEdit_hparams import AlphaEditHyperParams
from util import nethook
from util.globals import *
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from transformers.generation.streamers import TextIteratorStreamer

from PIL import Image

import requests
from io import BytesIO

from cog import BasePredictor, Input, Path, ConcatenateIterator
# import time
import subprocess
from threading import Thread

import os

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    "Ours": (MEMITHyperParams, apply_ours_to_model),
    "DINM": (DINMHyperParams, apply_dinm_to_model),
    "Ours_bp": (DINMHyperParams, apply_bp_to_model),
    "Ours_bp_cat": (DINMHyperParams, apply_bp_cat_to_model),
    "Ours_abs": (DINMHyperParams, apply_abs_to_model),
    "alphaedit": (AlphaEditHyperParams, apply_AlphaEdit_to_model)
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "privacy": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "science_qa": (ScienceQA, compute_rewrite_quality_counterfact),
    "mllmguard": (MLLMGUARD, )
}

device_map = {
    "model.vision_tower": 0,              # 视觉编码器在 GPU 0
    "model.mm_projector": 0,               # 投影层在 GPU 0
    "model.language_model.model.embed_tokens": 1,
    "model.language_model.model.layers.0": 1,
    # ... 中间层分布在 GPU 1-2
    "model.language_model.model.layers.24": 2,
    "model.language_model.model.layers.25": 2,
    "model.language_model.model.layers.26": 2,
    "model.language_model.model.norm": 3,  # 归一化层在 GPU 3
    "model.language_model.lm_head": 3      # 输出层在 GPU 3
}

class LLAVA(nn.Module):
    def __init__(self, args, device_map=None):
        super(LLAVA, self).__init__()

        # ========================================
        #             Model Initialization
        # ========================================
        self.model_name = "llava1.5-7b"
        # self.device = 'cuda:{}'.format(args.gpu_id)
        random_number = random.randint(1, 2000)
        random.seed(random_number)
        np.random.seed(random_number)
        torch.manual_seed(random_number)

        cudnn.benchmark = False
        cudnn.deterministic = True

        print('Initializing Chat')
        # cfg = Config(args)

        self.device = 'cuda'
        self.device_map = "cuda"
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model("/mnt/sdb1/caoxiangkui/LVLMs/models/LLaVA/checkpoints/llava-v1.5-7b",
                                                                               model_name="llava-v1.5-7b", device_map=self.device_map if device_map is None else device_map,
                                                                               model_base=None, load_8bit=False,
                                                                               load_4bit=False)
        print(self.model)
        # self.model = self.slice_model(self.model, 16)
        # model_config = cfg.model_cfg
        # model_config.device_8bit = args.gpu_id
        # model_cls = registry.get_model_class(model_config.arch)
        # self.model = self.model.to(self.device)

        self.model = self.model.eval()
        # self.print_device(self.model)
        # self.print_device(self.tokenizer)
        # self.print_device(self.image_processor)
        # self.model = self.slice_model(self.model, 17)

        # CONV_VISION = conv_dict[model_config.model_type]

        # stop_words_ids = [[835], [2277, 29937]]
        # stop_words_ids = [torch.tensor(ids).to(self.device) for ids in stop_words_ids]
        # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        # if stopping_criteria is not None:
        #     self.stopping_criteria = stopping_criteria
        # else:
        #     stop_words_ids = [torch.tensor([2]).to(self.device)]
        #     self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        # print('Initialization Finished')

        conv_mode = "llava_v1"
        self.conv = conv_templates[conv_mode].copy()
        # self.conv = CONV_VISION.copy()
        # self.conv.append_message(self.conv.roles[0], "<Img><ImageHere></Img>")


    def get_conv(self, inputs, outputs):
        output = []
        for inp, outp in zip(inputs, outputs):
            conv = self.conv.copy()
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], outp)
            output.append(conv)
        return output

    def get_images(self, paths):
        outputs = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            # image = self.image_transform([224, 224], mean, std)(image).unsqueeze(0).to(self.device)
            image = self.image_processor.preprocess(image, return_tensors='pt')[
                'pixel_values'].half().to(self.device)
            # print(f"llava image shape: {image.shape}")
            outputs.append(image)
        return outputs

    def get_pil(self, imgs):
        imgs = [img.convert('RGB') for img in imgs]
        imgs = [self.image_processor.preprocess(img, return_tensors='pt')[
                'pixel_values'].half().to(self.device) for img in imgs]
        return imgs

    def get_context_emb(self, conv, img):
        # print(img)
        # print("--------------------get_context_emb start----------------------------")
        if isinstance(img, list):
            assert len(img) == 1
            img = img[0]
        if isinstance(img, str):
            image_features = self.model.encode_images(img)
        else:
            image_features = img
        # print(f"image_features shape: {image_features.shape}")
        prompt = conv.get_prompt()
        # print(f"from conv prompt: {prompt}")
        '''
        #llama-2
        if flag==True:
            #print(prompt_segs)
            prompt_segs[1] = prompt_segs[1][:-3]
        '''
        # print(prompt_segs)
        # seg_tokens = [
        #     self.model.llama_tokenizer(
        #         seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
        #     # only add bos to the first seg
        #     for i, seg in enumerate(prompt_segs)
        # ]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(self.model.device)
        if input_ids[-1] == 2:
            input_ids = input_ids[:-1]

        # print(f"get_context_emb input_ids: {input_ids}")
        # print(f"get_context_emb input_ids: {self.tokenizer.decode(input_ids[-15:], skip_special_tokens=False)}")
        # prompt_chunks = self.tokenizer(prompt).input_ids
        # input_decode = self.tokenizer.decode(prompt_chunks)
        # print(f"input_ids shape: {input_ids.shape}")
        # print(f"input_decode: {input_decode}")

        image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        input_ids_img_pre = input_ids[:image_token_indices]
        input_ids_img_suff = input_ids[image_token_indices+1:]
        input_ids_with_img = torch.cat([input_ids_img_pre, torch.full([image_features.shape[1]], IMAGE_TOKEN_INDEX, device=input_ids_img_pre.device),input_ids_img_suff]).unsqueeze(0)
        # print(f"input_ids_with_img shape: {input_ids_with_img.shape}")

        input_embs_noimg = self.model.get_model().embed_tokens(torch.cat([input_ids_img_pre, input_ids_img_suff]))
        split_sizes = [input_ids_img_pre.shape[0], input_ids_img_suff.shape[0]]
        input_embs_img_pre, input_embs_img_suff = torch.split(input_embs_noimg, split_sizes, dim=0)
        # print(f"input_embs_img_pre shape: {input_embs_img_pre.shape}")
        # print(f"input_embs_img_suff shape: {input_embs_img_suff.shape}")
        image_features = image_features[0]
        input_embs_with_img = [input_embs_img_pre, image_features, input_embs_img_suff]
        input_embs_with_img = torch.cat(input_embs_with_img).unsqueeze(0)
        # print(f"input_embs_with_img shape: {input_embs_with_img.shape}")
        # print("--------------------get_context_emb end----------------------------")
        return input_embs_with_img, input_ids_with_img

    def find_fact_lookup_idx(self, prompt_no_subject, subject, tok, fact_token_strategy, verbose=True,):
        # print("---------------find_fact_lookup_idx start-----------------------------")
        assert prompt_no_subject.count("{}") == 1
        Image_toks = 24 * 24
        prompt = prompt_no_subject.replace(DEFAULT_IMAGE_TOKEN, "")
        # print(fact_token_strategy)
        ret = None
        if fact_token_strategy == "last":
            ret = -1
        elif (
                "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
        ):
            prompts = prompt.split("{}")
            assert prompts[0][-1] == " "
            prompts[0] = prompts[0][:-1]
            # full_index = prompt.index("{}")
            # prompts = [prompt[:full_index], prompt[full_index + 2:]]
            input_ids = [tok(chunk, add_special_tokens=i==0).input_ids for i, chunk in enumerate(prompts)]     #[[x,x,...,x,x], [x,x,...,x,x]]

            # print(f"input_ids_pre length: {len(input_ids[0])}")
            subject_ids = tok(subject, add_special_tokens=False).input_ids
            # subject_decode = tok.decode(subject_ids)
            # print(f"subject: {subject} | subject_ids length: {len(subject_ids)} | subject_decode: {subject_decode}")
            ret = len(input_ids[0]) + len(subject_ids) - 1
            # print(f"{ret} = {len(input_ids[0])} + {len(subject_ids)} - 1")
            # ret = repr_tools.get_words_idxs_in_templates(
            #     tok=tok,
            #     context_templates=[prompt],
            #     words=[subject],
            #     subtoken=fact_token_strategy[len("subject_"):],
            # )[0][0]
        else:
            raise ValueError(f"fact_token={fact_token_strategy} not recognized")

        sentence = prompt.format(subject)
        if verbose:
            # print(f"pre decode: {tok.decode(tok(sentence, return_tensors='pt', add_special_tokens=True).input_ids[0, :len(input_ids[0])])}")
            print(
                f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
                tok.decode(tok(sentence, return_tensors="pt", add_special_tokens=True).input_ids[0, ret]),
            )
        # print("---------------find_fact_lookup_idx end-----------------------------")
        return ret + Image_toks + 1 if ret != -1 else -1

    def get_conv_embedding_with_image(self, tok, imgs, inputs, outputs):
        # print("-----------get_conv_embedding_with_image start--------------------")
        if len(imgs) == 1:
            imgs = imgs * len(inputs)
        else:
            assert len(imgs) == len(inputs) == len(outputs)
        input_emb_with_img_all = []
        input_ids_with_img_all = []
        input_max_len = 0
        for img, inp, outp in zip(imgs, inputs, outputs):
            conv_t = self.get_conv([inp], [outp])[0]
            input_emb_with_img, input_ids_with_img = self.get_context_emb(conv_t, img)
            input_emb_with_img_all.append(input_emb_with_img)
            input_ids_with_img_all.append(input_ids_with_img)
            if input_ids_with_img.shape[1] > input_max_len:
                input_max_len = input_ids_with_img.shape[1]

        input_emb_with_img_tensor = []
        input_ids_with_img_tensor = []
        attention_mask = torch.zeros((len(inputs), input_max_len), dtype=input_ids_with_img.dtype, device=input_ids_with_img.device)

        for i, (input_emb_with_img, input_ids_with_img) in enumerate(zip(input_emb_with_img_all, input_ids_with_img_all)):
            input_cur_len = input_ids_with_img.shape[1]
            input_emb_with_img_tensor.append(torch.cat((
                input_emb_with_img,
                torch.zeros((1, input_max_len - input_cur_len, input_emb_with_img.shape[2]), dtype=input_emb_with_img.dtype,
                            device=input_emb_with_img.device),
            ), dim=1))
            input_ids_with_img_tensor.append(torch.cat((
                input_ids_with_img,
                torch.zeros((1, input_max_len - input_cur_len),
                            dtype=input_ids_with_img.dtype,
                            device=input_ids_with_img.device),
            ), dim=1))
            attention_mask[i, :input_cur_len] = 1
        input_emb_with_img_tensor = torch.cat(input_emb_with_img_tensor, dim=0)
        input_ids_with_img_tensor = torch.cat(input_ids_with_img_tensor, dim=0)
        # print(f"input_ids_with_img_tensor shape: {input_ids_with_img_tensor.shape}")
        # print(f"input_emb_with_img_tensor shape: {input_emb_with_img_tensor.shape}")
        # print(f"attention_mask shape: {attention_mask.shape}")
        # print("------------------get_conv_embedding_with_image end---------------------")
        return input_ids_with_img_tensor, attention_mask, input_emb_with_img_tensor

    def create_device_map(self, num_layers=32, num_gpus=4):
        """为LLaVA模型创建设备映射"""
        device_map = {}

        # 视觉部分放在 GPU 0
        # device_map["vision_tower"] = "cuda:0"
        device_map["vision_model"] = "cuda:0"
        device_map["model.mm_projector"] = "cuda:0"

        # 语言模型嵌入层放在 GPU 0
        device_map["model.embed_tokens"] = "cuda:0"

        # 计算每张卡分配的层数
        layers_per_gpu = num_layers // num_gpus
        remainder = num_layers % num_gpus

        # 分配模型层
        start_index = 0
        for gpu_id in range(0, num_gpus):
            # 计算当前GPU分配的层数
            if gpu_id < remainder:
                end_index = start_index + layers_per_gpu + 1
            else:
                end_index = start_index + layers_per_gpu

            # 为当前GPU分配层
            for layer_idx in range(start_index, end_index):
                device_map[f"model.layers.{layer_idx}"] = f"cuda:{gpu_id}"

            start_index = end_index

        # 归一化层和输出层放在最后的 GPU 上
        device_map["model.norm"] = f"cuda:{num_gpus - 1}"
        device_map["lm_head"] = f"cuda:{num_gpus - 1}"

        return device_map

    def print_device(self, model):
        print(model)

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
    questions_per_img:int,
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
    repetition=None,
    kl_factor=None
):
    # Set algorithm-specific variables
    if alg_name not in ["In-context", "Org"]:
        params_class, apply_algo = ALG_DICT[alg_name]
    if args.topk is not None and "Ours" in alg_name:
        topk = args.topk
    else:
        topk = None
    # Get run hyperparameters
    if alg_name not in ["In-context", "Org"]:
        params_path = HPARAMS_DIR / alg_name / hparams_fname
        print(f"params_path: {params_path}")

        hparams = params_class.from_json(params_path)
        if layer_check is not None:
            if isinstance(layer_check, int):
                hparams.layers = [layer_check]
            else:
                hparams.layers = layer_check
        print(f"Executing {alg_name} with parameters {hparams}")
        if kl_factor is not None:
            hparams.kl_factor = kl_factor

    model = LLAVA(args)
    print(model.model)
    tok = model.tokenizer
    # state_dict = torch.load("SKU/llava_unlearned_danger/llava7b_org_memflex.pt", map_location="cpu")
    # state_dict = torch.load("SKU/llava_unlearned_benign/llava1.5-7b_org_benign.pt", map_location="cpu")
    # model.model.load_state_dict(state_dict)
    # del state_dict
    if alg_name not in ["In-context", "Org"]:
        weights_copy = {
            f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
                model.model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            ).detach().clone()
            for layer in hparams.layers
        }
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
    cache_template = None
    # if use_cache:
    #     cache_template = (
    #         KV_DIR
    #         / f"{model_name.replace('/', '_')}_{alg_name}"
    #         / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
    #     )
    #     print(f"Will load cache from {cache_template}")

    # Iterate through dataset
    etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()
    privacy_value_dict = dict(privacy_value=args.privacy_value) if alg_name in ["Ours", "Ours_bp", "Ours_bp_cat"] else dict()
    topk_dict = dict(topk=topk) if topk is not None else dict()
    args_conserve_memory = (
        dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
        if conserve_memory
        else dict()
    )
    # hparams.rewrite_module_tmp = "model.layers.{}.self_attn.o_proj"
    print(f"num_edit: {num_edits}")
    edited_model = model
    if alg_name in ["DINM", "Ours_bp", "Ours_abs"]:
        num_edits = 1

    # return
    # # Evaluate new model
    edited_model.model.eval()
    model.model.eval()
    alg_name = "org_2new_cat"
    pope = False
    mme = False
    science_qa = False
    mllmguard = False

    ds_eval = ds_class(DATA_DIR, model=model, tok=tok, size=None, files=dataset_files)
    results = []
    for case in chunks(ds_eval, 1):
        record = case[0]
        img_paths = [os.path.join(IMAGE_ROOT, record["image"])]
        imgs = model.get_images(img_paths)
        # encode_img outputs tuple()
        if model.model_name == "llava1.5-7b":
            img_embs = [model.model.encode_images(img) for img in imgs]
        else:
            img_embs = [model.model.encode_img(img)[0][0].unsqueeze(0) for img in imgs]
        # img_embs = [model.model.encode_img(img)[0][0].unsqueeze(0) for img in imgs]
        if alg_name in ["In-context"]:
            incontext = ("You are a helpful AI assistant. "
                         "Your output should be beneficial, helpful, and insightful. "
                         "Please avoid generating sensitive or harmful content. "
                         "You should decline unreasonable user requests, such as those for pornographic or violent content.\n")
            record["paraphrase_prompts"] = [incontext + x for x in record["paraphrase_prompts"]]
            record["neighborhood_prompts"] = [incontext + x for x in record["neighborhood_prompts"]]
        metrics = {
            "case_id": record["case_id"],
            "num_edits": num_edits,
            "requested_rewrite": record["requested_rewrite"],
            "original question": generate_fast(edited_model, tok, img_embs,  [record["requested_rewrite"]["prompt"].replace("{}", record["requested_rewrite"]["subject"])], 50),
            "paraphrase_prompts": generate_fast(edited_model, tok, img_embs, record["paraphrase_prompts"], 50),
            "neighborhood_prompts": generate_fast(edited_model, tok, img_embs, record["neighborhood_prompts"], 50),
            # "generation_prompts": generate_fast(edited_model, tok, img_embs, record["generation_prompts"], 50),
        }
        results.append(metrics)
        print(f"case {record['case_id']} finished...")
    #
    # Dump metrics in .json
    if not os.path.exists("results"):
        os.mkdir("results")
    if privacy_scale:
        with open(os.path.join("results", model.model_name + alg_name + f"{f'_{repetition}' if repetition is not None else ''}" + f"{f'_{questions_per_img}' if alg_name == 'Ours_bp_cat' else ''}" + f"{f'_{hparams.kl_factor}' if alg_name != 'In-context' else ''}" + "_largedset_" + f"_{args.privacy_value}_{'single' if answer else 'multi'}{f'_{topk}' if topk else ''}_ans_edited_{privacy_scale}_{dataset_size_limit}_{num_edits}"
                                                     f"_with_tanh{'_'+f'{layer_check}' if layer_check is not None else ''}.json"), "w") as f:
            json.dump(results, f, indent=1)
    else:
        with open(os.path.join("results", model.model_name + alg_name + f"{f'_{repetition}' if repetition is not None else ''}" + "_with_tmps" + f"{f'_{questions_per_img}' if alg_name == 'Ours_bp_cat' else ''}" + f"{f'_{hparams.kl_factor}' if alg_name != 'In-context' else ''}" + "_largedset_" + f"_{args.privacy_value}_{'single' if answer else 'multi'}{f'_{topk}' if topk else ''}_ans_edited_{dataset_size_limit}_{num_edits}"
                                                   f"_with_tanh{'_'+f'{layer_check}' if layer_check is not None else ''}.json"), "w") as f:
            json.dump(results, f, indent=1)


    if science_qa:
        ascii_A = ord('A')
        science_qa_ds = ScienceQA(size=400)
        science_qa_res = []
        for case_id, case in enumerate(chunks(science_qa_ds, 1)):
            record = case[0]
            # img_paths = [os.path.join(IMAGE_ROOT, record["image"])]
            imgs = Image.open(BytesIO(record["image"]["bytes"]))
            imgs = model.get_pil([imgs])
            # encode_img outputs tuple()
            if model.model_name == "llava1.5-7b":
                img_embs = [model.model.encode_images(img) for img in imgs]
            else:
                img_embs = [model.model.encode_img(img)[0][0].unsqueeze(0) for img in imgs]
            # img_embs = [model.model.encode_img(img)[0][0].unsqueeze(0) for img in imgs]
            choice_str = ''.join([f"{chr(ascii_A + i)}: {c}\n" for i, c in enumerate(record['choices'])])
            prompt_scienceqa = [f"Question:{record['question']}\nChoices:\n{choice_str}"]
            if alg_name in ["In-context"]:
                incontext = ("You are a helpful AI assistant. "
                             "Your output should be beneficial, helpful, and insightful. "
                             "Please avoid generating sensitive or harmful content. "
                             "You should decline unreasonable user requests, such as those for pornographic or violent content.\n")
                prompt_scienceqa = [incontext + prompt_scienceqa[0]]
            metrics = {
                "case_id": case_id,
                "num_edits": num_edits,
                "question": record["question"],
                "choices": record["choices"],
                "outputs": generate_fast(edited_model, tok, img_embs, prompt_scienceqa, 200),
            }
            science_qa_res.append(metrics)
            print(f"case {case_id} finished...")

        # Dump metrics in .json
        if not os.path.exists("results"):
            os.mkdir("results")
        if privacy_scale:
            with open(os.path.join("results",
                                   f"{model.model_name}_ScienceQA" + alg_name + f"{f'_{repetition}' if repetition is not None else ''}" + f"{f'_{hparams.kl_factor}' if alg_name != 'In-context' else ''}" + f"_0_{'single' if answer else 'multi'}_ans_edited_{privacy_scale}_{dataset_size_limit}_{num_edits}"
                                              f"_with_tanh{'_' + f'{layer_check}' if layer_check else ''}.json"),
                      "w") as f:
                json.dump(science_qa_res, f, indent=1)
        else:
            with open(os.path.join("results",
                                   f"{model.model_name}_ScienceQA" + alg_name + f"{f'_{repetition}' if repetition is not None else ''}" + f"{f'_{hparams.kl_factor}' if alg_name != 'In-context' else ''}" + f"_0_{'single' if answer else 'multi'}_ans_edited_{dataset_size_limit}_{num_edits}"
                                              f"_with_tanh{'_' + f'{layer_check}' if layer_check else ''}.json"),
                      "w") as f:
                json.dump(science_qa_res, f, indent=1)
        # with open(os.path.join("results", f"ScienceQA_llava_before{f'_{repetition}' if repetition is not None else ''}.json"), "w") as f:
        #     json.dump(science_qa_res, f, indent=1)

    if mllmguard:
        mllmguard_qa_ds = MLLMGUARD(size=None)
        mllmguard_qa_res = []
        for case_id, case in enumerate(chunks(mllmguard_qa_ds, 1)):
            record = case[0]
            # img_paths = [os.path.join(IMAGE_ROOT, record["image"])]
            img_paths = [record["image"]]
            imgs = model.get_images(img_paths)
            # encode_img outputs tuple()
            if model.model_name == "llava1.5-7b":
                img_embs = [model.model.encode_images(img) for img in imgs]
            else:
                img_embs = [model.model.encode_img(img)[0][0].unsqueeze(0) for img in imgs]

            if alg_name in ["In-context"]:
                incontext = ("You are a helpful AI assistant. "
                             "Your output should be beneficial, helpful, and insightful. "
                             "Please avoid generating sensitive or harmful content. "
                             "You should decline unreasonable user requests, such as those for pornographic or violent content.\n")
                record["prompt"] = incontext + record["prompt"]
            metrics = {
                "case_id": case_id,
                "num_edits": num_edits,
                "question": record["prompt"],
                "outputs": generate_fast(edited_model, tok, img_embs, [record["prompt"]], 200),
            }
            mllmguard_qa_res.append(metrics)
            print(f"case {case_id} finished...")

        # Dump metrics in .json
        if not os.path.exists("results"):
            os.mkdir("results")
        if privacy_scale:
            with open(os.path.join("results",
                                   f"MLLMGuard_{model.model_name}_" + alg_name + f"{f'_{repetition}' if repetition is not None else ''}" + f"{f'_{hparams.kl_factor}' if alg_name != 'In-context' else ''}" + f"_0_{'multi' if answer else 'single'}_ans_edited_{privacy_scale}_{dataset_size_limit}_{num_edits}"
                                              f"_with_tanh{'_' + f'{layer_check}' if layer_check else ''}.json"),
                      "w") as f:
                json.dump(mllmguard_qa_res, f, indent=1)
        else:
            with open(os.path.join("results",
                                   f"MLLMGuard_{model.model_name}_" + alg_name + f"{f'_{repetition}' if repetition is not None else ''}" + f"{f'_{hparams.kl_factor}' if alg_name != 'In-context' else ''}" + f"_0_{'multi' if answer else 'single'}_ans_edited_{dataset_size_limit}_{num_edits}"
                                              f"_with_tanh{'_' + f'{layer_check}' if layer_check else ''}.json"),
                      "w") as f:
                json.dump(mllmguard_qa_res, f, indent=1)


    if pope:
        pope_dataset = POPE()
        pope_results = []
        for case_id, case in enumerate(chunks(pope_dataset, 1)):
            record = case[0]
            # img_paths = [os.path.join(IMAGE_ROOT, record["image"])]
            img_paths = [record["image"]]
            imgs = model.get_pil(img_paths)
            # encode_img outputs tuple()
            if model.model_name == "llava1.5-7b":
                img_embs = [model.model.encode_images(img) for img in imgs]
            else:
                img_embs = [model.model.encode_img(img)[0][0].unsqueeze(0) for img in imgs]

            if alg_name in ["In-context"]:
                incontext = ("You are a helpful AI assistant. "
                             "Your output should be beneficial, helpful, and insightful. "
                             "Please avoid generating sensitive or harmful content. "
                             "You should decline unreasonable user requests, such as those for pornographic or violent content.\n")
                record["prompt"] = incontext + record["prompt"]
            metrics = {
                "case_id": case_id,
                "num_edits": num_edits,
                "question": record["prompt"],
                "outputs": generate_fast(edited_model, tok, img_embs, [record["prompt"]], 200),
                "answer": record["answer"],
            }
            pope_results.append(metrics)
            print(f"case {case_id} finished...")

        # Dump metrics in .json
        if not os.path.exists("results"):
            os.mkdir("results")
        with open(os.path.join("results",
                               "POPE_" + model.model_name + f"_{alg_name}_{kl_factor}_" + f"{num_edits}_" + f"{f'{layer_check}' if layer_check else ''}" + "_largedset_440" + ".json"),
                  "w") as f:
            json.dump(pope_results, f, indent=1)

    if mme:
        mme_dataset = MME_Dataset()
        mme_results = []
        for case_id, case in enumerate(chunks(mme_dataset, 1)):
            image_path = case[0]
            assert ".jpg" in image_path or ".png" in image_path, image_path
            if "/image/" in image_path or "/images/" in image_path:
                text_path = image_path.replace(".jpg", ".txt") if ".jpg" in image_path else image_path.replace(
                    ".png",
                    ".txt")
                text_path = text_path.replace("/image/", "/questions_answers_YN/")
                text_path = text_path.replace("/images/", "/questions_answers_YN/")
            else:
                text_path = image_path.replace(".jpg", ".txt") if ".jpg" in image_path else image_path.replace(
                    ".png", ".txt")
            questions = []
            with open(text_path, "r", encoding="utf-8") as f:
                for line in f:
                    arr = line.strip().split("\t")
                    assert len(arr) == 2
                    questions.append(arr)
            assert len(questions) == 2
            imgs = image_path
            imgs = model.get_images([imgs])
            if model.model_name == "llava1.5-7b":
                img_embs = [model.model.encode_images(img) for img in imgs]
            else:
                img_embs = [model.model.encode_img(img)[0][0].unsqueeze(0) for img in imgs]
            for i, q in enumerate(questions):
                question = f'{questions[i][0]}'
                if alg_name in ["In-context"]:
                    incontext = ("You are a helpful AI assistant. "
                                 "Your output should be beneficial, helpful, and insightful. "
                                 "Please avoid generating sensitive or harmful content. "
                                 "You should decline unreasonable user requests, such as those for pornographic or violent content.\n")
                    question = incontext + question
                metrics = {
                    "case_id": f"{case_id}-{i}",
                    "num_edits": num_edits,
                    "image path": image_path,
                    "question": question,
                    "outputs": generate_fast(edited_model, tok, img_embs, [question], 200),
                    "answer": questions[i][1],
                }
                mme_results.append(metrics)
            print(f"case {case_id} finished...")

        if not os.path.exists("results"):
            os.mkdir("results")
        with open(os.path.join("results",
                               "MME_" + model.model_name + f"_{alg_name}_{kl_factor}_" + f"{num_edits}_" + f"{f'{layer_check}' if layer_check else ''}" + "_largedset_440" + ".json"),
                  "w") as f:
            json.dump(mme_results, f, indent=1)
        # with open(os.path.join("results", "MLLMGuard_llava_before.json"), "w") as f:
        #     json.dump(mllmguard_qa_res, f, indent=1)

    print("Evaluation Finished!")


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
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME", "FT", "MEND", "Ours", "DINM", "Ours_bp", "Ours_abs", "Ours_bp_cat", "In-context", "alphaedit"],
        default="Ours",
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
        default="llava1.5-7b",
        help="Model to edit.",
        # required=True,
    )
    # parser.add_argument("--cfg-path", default='eval_configs/minigpt4_llama2_eval.yaml', help="path to configuration file.")
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
        default="llava1.5-7b.json",
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
        default=3,
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


    for alg, la in [("Ours_bp_cat", 11)]:
        for kl in [1.25]:
            main(
                alg,
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
                layer_check=la,
                science_qa=args.science_qa,
                mllmguard=args.mllmguard,
                kl_factor=kl
            )
