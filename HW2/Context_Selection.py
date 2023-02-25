from model_and_dataset import Muldataset,QAdataset,MultipleChoiceModel,QuestionAnsweringModel 
from accelerate import Accelerator
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
from argparse import ArgumentParser
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
def main(args):
    model_name='hfl/chinese-macbert-base'
    max_len=512
    config = AutoConfig.from_pretrained(model_name, return_dict=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config, model_max_length=max_len)
    
    mul_train=Muldataset(tokenizer=tokenizer,split='train',data_dir=args.data_dir)
    mul_valid=Muldataset(tokenizer=tokenizer,split='valid',data_dir=args.data_dir)
    
    device='cuda'
    
    batch_size=1
    train_loader = DataLoader(
            mul_train,
            collate_fn=mul_train.collate_fn,
            shuffle=True,
            batch_size=batch_size,
        )
    valid_loader = DataLoader(
            mul_valid,
            collate_fn=mul_valid.collate_fn,
            shuffle=True,
            batch_size=batch_size,
        )
        
    mul_model=MultipleChoiceModel(config,model_name).to(device)
    mul_optimizer = torch.optim.AdamW(
            mul_model.parameters(), lr=2e-5,weight_decay=1e-6
        )
        
    epochs=3
    best_acc=-1
    accelerator=Accelerator(fp16=True)
    accelerator.prepare(
            mul_model, mul_optimizer, train_loader, valid_loader
        )
    update_step=epochs*len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
            mul_optimizer, 0.1 * update_step, update_step
        )
    for epoch in range(epochs):
      mul_model.train()
      total=0
      accs=[]
      for batch in tqdm(train_loader):
        ids, input_ids, attention_masks, token_type_ids, labels=batch
        ids, input_ids, attention_masks, token_type_ids, labels=ids, input_ids.to(device), attention_masks.to(device), token_type_ids.to(device), labels.to(device)
        loss, logits = mul_model(
                input_ids=input_ids,
                # attention_mask=attention_masks,
                # token_type_ids=token_type_ids,
                labels=labels,
            )
        acc = (logits.argmax(dim=-1) == labels).cpu().float().mean()
        # print(acc)
        accs.append(acc)
        
        mul_optimizer.zero_grad()
        loss.backward()
        mul_optimizer.step()
        scheduler.step()
        total+=1
      print(sum(accs)/total)
      eval_total=0
      eval_accs=[]
      mul_model.eval()
      for batch in tqdm(valid_loader):
        ids, input_ids, attention_masks, token_type_ids, labels=batch
        ids, input_ids, attention_masks, token_type_ids, labels=ids, input_ids.to(device), attention_masks.to(device), token_type_ids.to(device), labels.to(device)
        loss, logits = mul_model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                token_type_ids=token_type_ids,
                labels=labels,
            )
        acc = (logits.argmax(dim=-1) == labels).cpu().float().mean()
        eval_accs.append(acc)
        eval_total+=1
      if sum(eval_accs)/eval_total>best_acc:
        best_acc=sum(eval_accs)/eval_total
        torch.save({'model_state_dict':mul_model.state_dict(),'opt':mul_optimizer.state_dict(),'sch':scheduler.state_dict()},f'../ADL_HW2/mul_{best_acc}.pt')
        print('best model saved')
        print(sum(eval_accs)/eval_total)
def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=Path, default="./")


    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
