from accelerate import Accelerator
from torch.utils import data
from model_and_dataset import Muldataset,QAdataset,MultipleChoiceModel,QuestionAnsweringModel
import json
import os
import random
random.seed(891015)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertConfig,
    get_cosine_schedule_with_warmup,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path
def main(args):
    model_name='hfl/chinese-macbert-base'
    max_len=512
    config = AutoConfig.from_pretrained(model_name, return_dict=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config, model_max_length=max_len)
    QA_train=QAdataset(tokenizer=tokenizer,split='train',data_dir=args.data_dir)
    QA_valid=QAdataset(tokenizer=tokenizer,split='valid',data_dir=args.data_dir)
    
    
    batch_size=4
    QA_train_loader = DataLoader(
            QA_train,
            collate_fn=QA_train.collate_fn,
            shuffle=True,
            batch_size=batch_size,
        )
    QA_valid_loader = DataLoader(
            QA_valid,
            collate_fn=QA_valid.collate_fn,
            shuffle=True,
            batch_size=batch_size,
        )
    device='cuda'
    QA_model=QuestionAnsweringModel(config,model_name).to(device)
    QA_optimizer = torch.optim.AdamW(
            QA_model.parameters(), lr=2e-5,weight_decay=1e-5
        )
    epochs=5
    update_step=epochs*len(QA_train_loader)
    
    train_loss = []
    train_accs = []
    # epochs=10
    best_acc=-1
    # log_step = 1
    # update_step=epochs*len(QA_train_loader)
    accelerator = Accelerator(fp16=True)
    accelerator.prepare(
            QA_model, QA_optimizer, QA_train_loader, QA_valid_loader
        )
    step=1
    for epoch in range(epochs):
      train_loss = []
      train_accs = []
    
      QA_model.train()
      for idx, batch in enumerate(tqdm(QA_train_loader)):
          inputs = batch
          input_ids = inputs['input_ids']
          start_positions = inputs['start_positions']
          end_positions = inputs['end_positions']
    
          loss,start_logits,end_logits = QA_model(
              input_ids=input_ids,
              start_positions=start_positions,
              end_positions=end_positions,
          )
          loss.backward()
          step+=1
          if step%2 ==0: 
            QA_optimizer.step()
            QA_optimizer.zero_grad()
          # scheduler.step()
    
    
          start_logits = start_logits.argmax(dim=-1)
          # print(start_positions)
          end_logits = end_logits.argmax(dim=-1)
          # print(end_positions)
          acc = (
              ((start_positions == start_logits) & (end_positions == end_logits))
              .cpu()
              .numpy()
              .mean()
          )
    
          train_loss.append(loss.item())
          train_accs.append(acc)
    
      train_loss = sum(train_loss) / len(train_loss)
      train_acc = sum(train_accs) / len(train_accs)
      print(train_acc)
    
      valid_loss = []
      valid_accs = []
    
      QA_model.eval()
      for idx, batch in enumerate(tqdm(QA_valid_loader)):
          inputs = batch
          input_ids = inputs['input_ids']
          start_positions = inputs['start_positions']
          end_positions = inputs['end_positions']
    
          loss,start_logits,end_logits = QA_model(
              input_ids=input_ids,
              start_positions=start_positions,
              end_positions=end_positions,
              # token_type_ids=inputs[3].to(device),
              # attention_mask=inputs[4].to(device)
          )
          # QA_optimizer.zero_grad()
          # loss.backward()
          # QA_optimizer.step()
    
          start_logits = start_logits.argmax(dim=-1)
          # print(start_positions)
          end_logits = end_logits.argmax(dim=-1)
          # print(end_positions)
          acc = (
              ((start_positions == start_logits) & (end_positions == end_logits))
              .cpu()
              .numpy()
              .mean()
          )
    
          valid_loss.append(loss.item())
          valid_accs.append(acc)
    
      valid_loss = sum(valid_loss) / len(valid_loss)
      valid_acc = sum(valid_accs) / len(valid_accs)
      if(valid_acc>best_acc):
        best_acc=valid_acc
        torch.save({'model_state_dict':QA_model.state_dict(),'opt':QA_optimizer.state_dict(),'sch':scheduler.state_dict()},f'../ADL_HW2/QA_{best_acc}.pt')
        print('best QA model saved')
    
      print(valid_acc)
def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=Path, default="./")
  

    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()
    main(args)