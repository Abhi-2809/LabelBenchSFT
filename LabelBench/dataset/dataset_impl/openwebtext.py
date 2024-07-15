import os
import torch
from torch.utils.data import Dataset, Subset
from datasets import load_dataset, load_from_disk

from LabelBench.dataset.dataset_skeleton import register_dataset, DataType, LabelType


def pad_to_len(tokens, tokenizer, length):
    if len(tokens) < length:
        if tokenizer.padding_side == "left":
            return [tokenizer.pad_token_id for _ in range(length - len(tokens))] + tokens
        else:
            return tokens + [tokenizer.pad_token_id for _ in range(length - len(tokens))]
    else:
        return tokens


class OpenWebTextDataset(Dataset):
    def __init__(self, tokenizer, data_dir="./data"):
        max_len = 1024
        if os.path.exists(os.path.join(data_dir, "openwebtext_chuncked")):
            dataset = load_from_disk(os.path.join(data_dir, "openwebtext_chuncked"))
        else:
            dataset = load_dataset("openwebtext", num_proc=12, split="train")

            def chunk_examples(examples):
                chunks = []
                texts = []
                for seg in examples["text"]:
                    output = tokenizer.tokenize(seg)
                    left = 0
                    right = min(max_len, len(output))
                    count = 0
                    while True:
                        count += 1
                        if right == len(output):
                            token_ids = tokenizer.convert_tokens_to_ids(output[left:right])
                            texts.append(tokenizer.decode(token_ids))
                            chunks.append(pad_to_len(token_ids, tokenizer, max_len))
                            break
                        elif output[right].startswith("▁") or left == right:
                            if left == right:
                                right = left + max_len
                            token_ids = tokenizer.convert_tokens_to_ids(output[left:right])
                            texts.append(tokenizer.decode(token_ids))
                            chunks.append(pad_to_len(token_ids, tokenizer, max_len))
                            left = right
                            right = min(left + max_len, len(output))
                        else:
                            right -= 1
                            assert right >= left, str(output[left:min(left + max_len, len(output))])
                return {"input_ids": chunks, "text": texts}

            dataset = dataset.map(chunk_examples, batched=True, remove_columns=dataset.column_names, num_proc=20)
            dataset.save_to_disk(os.path.join(data_dir, "openwebtext_chuncked"))
        self.text_dataset = dataset.remove_columns("input_ids")
        self.dataset = dataset.remove_columns("text")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[int(idx)]
        return torch.LongTensor(example["input_ids"]), idx


@register_dataset("openwebtext", DataType.TEXT, LabelType.MULTI_CLASS)
def get_openwebtext_dataset(data_dir, tokenizer):
    dataset = OpenWebTextDataset(tokenizer, data_dir=data_dir)
    return dataset, 2, None
