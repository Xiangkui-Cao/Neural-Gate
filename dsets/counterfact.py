import json
import typing
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *

from datasets import load_dataset

# REMOTE_ROOT = f"{REMOTE_ROOT_URL}/data/dsets"


class CounterFactDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        model,
        multi: bool = False,
        size: typing.Optional[int] = None,
        files = None,
        start_idx = None,
        *args,
        **kwargs,
    ):
        data_dir = Path(data_dir)
        if files is None:
            cf_loc = data_dir / (
                "neural_gate_documents_en.json"
            )
            with open(cf_loc, "r") as f:
                self.data = json.load(f)
            if size is not None:
                self.data = self.data[:size]

            print(f"Loaded dataset with {len(self)} elements")
        else:
            self.data = []
            if model.model_name == "llava1.5-7b":
                with open("results/llava1.5-7b_before.json", "r") as f: # original results of llava
                    gts_before = json.load(f)
                print(f"loading results/llava1.5-7b_before.json")
            else:
                with open("results/minigpt4-llama2-7b_before.json", "r") as f: # original results of minigpt
                    gts_before = json.load(f)
                print("loading results/minigpt4-llama2-7b_before.json")
            gts_before_select = []
            if len(files) == 1:
                cf_loc = data_dir / (
                    files[0]
                )
                with open(cf_loc, "r") as f:
                    data_t = json.load(f)
                if start_idx is None: # count data scale
                    self.data += data_t
                    gts_before_select = gts_before[:len(data_t)]
                else:
                    if size is not None:
                        if size < len(data_t) // 3:
                            self.data = self.data + data_t[:size]
                            gts_before_select = gts_before[start_idx:start_idx + size]
                        else:
                            self.data = self.data + data_t[:len(data_t) // 3]
                            gts_before_select = gts_before[start_idx:start_idx + len(data_t) // 3]
                    else:
                        self.data = self.data + data_t
                        gts_before_select = gts_before[start_idx:start_idx + len(data_t)]
            else:
                index_st_all = 0
                for file in files:
                    cf_loc = data_dir / (
                        file
                    )
                    with open(cf_loc, "r") as f:
                        data_t = json.load(f)
                    if size is not None:
                        if size < len(data_t)//3:
                            self.data = self.data + data_t[:size]
                            gts_before_select += gts_before[index_st_all:index_st_all+size]
                        else:
                            self.data = self.data + data_t[:len(data_t)//3]
                            gts_before_select += gts_before[index_st_all:index_st_all+len(data_t)//3]
                    else:
                        self.data = self.data + data_t
                        gts_before_select += gts_before[index_st_all:index_st_all+len(data_t)]
                    index_st_all += len(data_t)
            assert len(gts_before_select) == len(self.data)
            for gt, d in zip(gts_before_select, self.data):
            #     d["rewrite_answers"] = gt["original question"]
                d["paraphrase_answers"] = gt["paraphrase_prompts"]
                d["neighborhood_answers"] = gt["neighborhood_prompts"]

        if size is None:
            for i in range(len(self.data)):
                self.data[i]["case_id"] = i


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class MultiCounterFactDataset(CounterFactDataset):
    def __init__(
        self, data_dir: str, model, size: typing.Optional[int] = None, *args, **kwargs
    ):
        super().__init__(data_dir, model, *args, multi=True, size=size, **kwargs)

class ScienceQA(Dataset):
    def __init__(
        self,
        size: typing.Optional[int] = 100,
    ):
        data_dir = ""
        data = load_dataset("parquet", data_files={"test": data_dir}, split="test")
        self.data = [d for d in data if d["image"]]

        if size is not None:
            self.data = self.data[:size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class MLLMGUARD(Dataset):
    def __init__(
        self,
        size: typing.Optional[int] = 100,
    ):
        data_dir = "prompt.csv" # the path of mllmguard
        self.data = self.load_data(data_dir)

        if size is not None:
            self.data = self.data[:size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def read_csv(self, path):
        import csv
        with open(path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            data_list = [row for row in csv_reader]
        return data_list[108:] # Ignore the Chinese question

    def load_data(self, path):
        data = self.read_csv(path)
        image_root = ""
        data = [
            {
                "id": idx,
                "image": os.path.join(image_root, d[0]),
                "prompt": d[1]
            }
            for idx, d in enumerate(data)
        ]
        return data

def privacy_ds_cat(
        data_dir,
        model,
        size,
        files = None
):
    out_ds = []
    start_idx = 0
    data_size_list = count_data_size(data_dir, model, files)
    for file, data_size in zip(files, data_size_list):
        out_ds.append(CounterFactDataset(
            data_dir,
            model,
            size=size,
            files=[file],
            start_idx=start_idx
        ))
        start_idx += data_size
    return out_ds

def count_data_size(data_dir, model, files):
    size_out = []
    for file in files:
        size_out.append(CounterFactDataset(
            data_dir,
            model,
            files=[file],
        ).__len__())
    return size_out


class MME_Dataset(Dataset):
    def __init__(self, data_dir=""):
        dirs = os.listdir(data_dir)
        self.data = []
        for dir in dirs:
            if "image" in os.listdir(os.path.join(data_dir, dir)):
                images = os.listdir(os.path.join(data_dir, dir, "image"))
                image_paths = [os.path.join(data_dir, dir, "image", image) for image in images]
            elif "images" in os.listdir(os.path.join(data_dir, dir)):
                images = os.listdir(os.path.join(data_dir, dir, "images"))
                image_paths = [os.path.join(data_dir, dir, "images", image) for image in images]
            else:
                images = os.listdir(os.path.join(data_dir, dir))
                images = [image for image in images if ".txt" not in image]
                image_paths = [os.path.join(data_dir, dir, image) for image in images]
            self.data = self.data + image_paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class POPE(Dataset):
    def __init__(self, data_dir = ""):
        # data = load_dataset("parquet", data_files={"test": data_dir}, config_name="Full", split="random")
        # data = load_dataset(path=data_dir, config_name="Full", split="random")
        dataset = load_dataset(
            "parquet",
            data_files={
                "random": f"{data_dir}/test-*.parquet"
            },
            split="random"
        )
        self.data = [d for d in dataset if d["category"]=='random']
        # self.data = [d for d in data if d["image"]]

        # if size is not None:
        #     self.data = self.data[:size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # print(sample)
        return {
            "id": sample["id"],
            "image": sample["image"],
            "prompt": f"{sample['question']} Please answer yes or no.",
            "answer": sample["answer"],
        }

