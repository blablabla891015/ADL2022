from typing import List, Dict
import torch
from torch.utils.data import Dataset

from utils import Vocab,pad_to_len
import re
class SeqClsDataset(Dataset):
    def __init__(
        self,
        
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        mode:str,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.mode=mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        device='cuda'
        text,intent,id=[],[],[]
        for sample in samples:
            id.append(sample['id'])
            if self.mode!='test':
                intent.append(self.label_mapping[sample['intent']])
            text.append(re.sub(r'[^\w\s]', '', sample["text"]).split())
            
        text=self.vocab.encode_batch(text,self.max_len)
        if self.mode!='test':
            return torch.tensor(text).to(device),torch.tensor(intent).to(device),id
        else:
            return torch.tensor(text).to(device),id

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    # ignore_idx = -100
    def collate_fn(self, samples):
        device='cuda'
        tokens, tags, length, id = [], [], [], []
        for sample in samples:
            id.append(sample["id"])
            if self.mode != "test":
                tags.append([self.label_mapping[tag] for tag in sample["tags"]])
            length.append(len(sample["tokens"]))
            tokens.append(sample["tokens"])
        tokens = self.vocab.encode_batch(tokens, self.max_len)
        tokens = torch.tensor(tokens).to(device)
        if self.mode != "test":
            tags = pad_to_len(tags, self.max_len, 9)
            tags = torch.tensor(tags).to(device)
            # print(tags)
            return tokens, tags, length, id
        return tokens, length, id
            
