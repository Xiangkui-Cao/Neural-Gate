import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from minigpt4.conversation.conversation import CONV_VISION_LLama2

class TokenizedDataset(Dataset):
    """
    Converts a dataset of text samples into a dataset of token sequences,
    as converted by a supplied tokenizer. The tokens come along with position
    ids and attention masks, they can be supplied direcly to the model.
    """

    def __init__(self, text_dataset, model, tokenizer=None, maxlen=None, field="text"):
        self.text_dataset = [x for x in text_dataset if x[field] != "" and len(x[field]) <= maxlen]
        self.field = field
        self.model_name = model.model_name
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        if hasattr(text_dataset, "info"):
            self.info = text_dataset.info

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, i):
        text = self.text_dataset[i]
        if self.field is not None:
            text = text[self.field]
        conv = CONV_VISION_LLama2.copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], "")
        text = conv.get_prompt()
        if self.model_name == "llava1.5-7b":
            token_list = self.tokenizer(text, add_special_tokens=True).input_ids
            token_list = torch.tensor(token_list, dtype=torch.long).unsqueeze(0)
        else:
            token_list = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
        return token_list


def dict_to_(data, device):
    """
    Moves a dictionary of tensors to the specified device.
    """
    for k in data:
        data[k] = data[k].to(device)
    return data


def length_collation(token_size):
    """
    Sorts a batch of sequences and breaks it up into subbatches
    of same-sized sequences, padding as needed.  Each batch
    has no more than token_size total tokens (or a single
    sequence, if the sequence happens to be larger).
    """

    def collate_fn(items):
        items = sorted(items, key=lambda x: -len(x["input_ids"]))
        batches = []
        batch = []
        batch_width = 0
        for item in items:
            item_width = len(item["input_ids"])
            if item_width == 0:
                break
            if batch_width * (len(batch) + 1) > token_size:
                batches.append(make_padded_batch(batch))
                batch = []
                batch_width = 0
            if not batch:
                batch_width = item_width
            batch.append(item)
        if len(batch):
            batches.append(make_padded_batch(batch))
        return batches

    return collate_fn


def make_padded_batch(items):
    """
    Pads sequences in a batch, so they are all the same length as the longest.
    """
    max_len = max(len(d["input_ids"]) for d in items)
    if max_len == 0:
        return {k: torch.zeros((0, 0), dtype=torch.long) for k in items[0]}
    return {
        k: pad_sequence([d[k] for d in items if len(d["input_ids"])], batch_first=True)
        for k, v in items[0].items()
    }


def flatten_masked_batch(data, mask):
    """
    Flattens feature data, ignoring items that are masked out of attention.
    """
    flat_data = data.view(-1, data.size(-1))
    attended_tokens = mask.view(-1).nonzero()[:, 0]
    return flat_data[attended_tokens]
