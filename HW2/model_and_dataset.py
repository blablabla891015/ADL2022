from torch.utils import data
import json
import os
import random
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertConfig,
    get_cosine_schedule_with_warmup,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMultipleChoice, AutoModelForQuestionAnswering


class Muldataset(Dataset):
  def __init__(self,tokenizer,split,data_dir='./',context_path='./contest.json',data_path='./test.json'):
    self.tokenized_data=[]
    if context_path != './contest.json':
        pass
    else:
        context_path=os.path.join(data_dir,'context.json')
    context_file=json.load(open(context_path,encoding="utf-8"))
    self.split=split
    if data_path != './test.json':
        pass
    else:
        data_path=os.path.join(data_dir,f'{split}.json')
    data_file=json.load(open(data_path,encoding="utf-8"))
    for dic in tqdm(data_file):
      if split !='test':
        label = dic["paragraphs"].index(dic["relevant"])
      id=dic['id']
      qa_pair = ["{} {}".format(dic["question"], context_file[i])for i in dic["paragraphs"]]
      feature = tokenizer(qa_pair,padding="max_length",truncation=True,return_tensors="pt",)
      # print(feature['input_ids'])
      if split !='test':
          self.tokenized_data.append(
          {'id':id,'label':label,'input_ids':feature['input_ids']
          ,'token_type_ids':feature['token_type_ids'],'attention_mask':feature['attention_mask']}
      )
      else:
          self.tokenized_data.append(
          {'id':id,'label':[],'input_ids':feature['input_ids']
          ,'token_type_ids':feature['token_type_ids'],'attention_mask':feature['attention_mask']}
      )
        
  def __len__(self):
      return len(self.tokenized_data)

  def __getitem__(self, idx):
      return self.tokenized_data[idx]
  def collate_fn(self, batch):
      ids, input_ids, attention_masks, token_type_ids, labels = [], [], [], [], []
      for sample in batch:
        ids.append(sample["id"])
        input_ids.append(sample["input_ids"])
        token_type_ids.append(sample["token_type_ids"])
        attention_masks.append(sample["attention_mask"])
        labels.append(sample['label'])
      input_ids = torch.stack(input_ids)
      attention_masks = torch.stack(attention_masks)
      token_type_ids = torch.stack(token_type_ids)
      labels = torch.LongTensor(labels)
      return ids, input_ids, attention_masks, token_type_ids, labels
class QAdataset(Dataset):
  def __init__(self,tokenizer,split,data_dir='./',pre_label=None,context_path='./contest.json',data_path='./test.json'):
    self.datas=[]
    if context_path != './contest.json':
        pass
    else:
        context_path=os.path.join(data_dir,'context.json')
    context_file=json.load(open(context_path,encoding="utf-8"))
    if data_path != './test.json':
        pass
    else:
        data_path=os.path.join(data_dir,f'{split}.json')
    data_file=json.load(open(data_path,encoding="utf-8"))
    if pre_label != None and split=='test':
        pre_label_file=json.load(open(pre_label,encoding="utf-8"))
    self.tokenizer=tokenizer
    self.split=split
    for dic in tqdm(data_file):
      data = {
          "id": dic["id"],
          "question": dic["question"],
      }
      if split != "test":
        data.update(
            {
              "context": context_file[dic["relevant"]],
              "answer": dic["answer"],
            }
        )
      else:
        data.update(
            {
              "context": context_file[pre_label_file[dic['id']]],
            }
        )
        
      self.datas.append(data)
  def __len__(self):
      return len(self.datas)

  def __getitem__(self, idx):
      return self.datas[idx]
  def collate_fn(self, batch):
    device='cuda'
    ids = [sample["id"] for sample in batch]
    inputs = self.tokenizer(
        [data["question"] for data in batch],
        [data["context"] for data in batch],
        truncation="only_second",
        stride=128,  # TODO: change to 32 for model from scatch
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
    # inputs["context"] = [data["context"] for data in batch]

    if self.split != "test":
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            start_char = batch[sample_idx]["answer"]["start"]
            end_char = batch[sample_idx]["answer"]["start"] + len(
                batch[sample_idx]["answer"]["text"]
            )
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if (
                offset[context_start][0] > start_char
                or offset[context_end][1] < end_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = torch.tensor(start_positions).to(device)
        inputs["end_positions"] = torch.tensor(end_positions).to(device)
        inputs['input_ids']=inputs['input_ids'].to(device)
    else:
        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []
        offset_mapping = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(batch[sample_idx]["id"])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            offset_mapping.append(
                [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]
            )

        inputs["example_id"] = example_ids
        inputs["offset_mapping"] = offset_mapping
        inputs['context']=[data["context"] for data in batch]
    return inputs
class MultipleChoiceModel(nn.Module):
    def __init__(self,config, name=None,if_ckpt=False):
        super(MultipleChoiceModel, self).__init__()
        self.name = name
        if if_ckpt:
            self.model = AutoModelForMultipleChoice.from_config(config)
        else:
            self.model = AutoModelForMultipleChoice.from_pretrained(self.name, config = config)
            

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def freeze_bert(self):
        print("Freezing BERT")
        for param in self.model.bert.parameters():
            param.requires_grad = False
class QuestionAnsweringModel(nn.Module):
    def __init__(self,config, name=None,if_ckpt=False):
        super(QuestionAnsweringModel, self).__init__()
        self.name = name
        if if_ckpt:
            self.model=self.model = AutoModelForQuestionAnswering.from_config(config)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.name,config = config)


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def freeze_bert(self):
        print("Freezing BERT")
        for param in self.model.bert.parameters():
            param.requires_grad = False
