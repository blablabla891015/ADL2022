import os
from model_and_dataset import Muldataset,QAdataset,MultipleChoiceModel,QuestionAnsweringModel
from accelerate import Accelerator
from torch.utils import data
import json
import os
import random
from argparse import ArgumentParser
from pathlib import Path
from torch.utils.data import DataLoader
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

def main(args):
  device='cuda'
  model_name='hfl/chinese-macbert-base'
  max_len=512
  config = AutoConfig.from_pretrained(pretrained_model_name_or_path='./config.json', return_dict=False)
  tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./tokenizer/', model_max_length=max_len)
  mul_test=Muldataset(tokenizer=tokenizer,split='test',context_path=args.context_path,data_path=args.test_path)
  batch_size=1
  test_loader = DataLoader(
          mul_test,
          collate_fn=mul_test.collate_fn,
          shuffle=False,
          batch_size=batch_size,
      )
  ckpt = torch.load(args.mc_ckpt)
  mul_model=MultipleChoiceModel(config,model_name,if_ckpt=True).to(device)
  mul_model.load_state_dict(ckpt['model_state_dict'])
  mul_model.eval()
  res={}
  for batch in tqdm(test_loader):
      ids, input_ids, attention_masks, token_type_ids, _=batch
      ids, input_ids, attention_masks, token_type_ids=ids, input_ids.to(device), attention_masks.to(device), token_type_ids.to(device)
      logits = mul_model(
              input_ids=input_ids,
          )
      
      res[ids[0]]=int(logits[0].argmax(dim=-1))
  test_file=json.load(open(args.test_path,encoding="utf-8"))
  with open('./pre_label.json','w') as f:
    dics={}
    for i in test_file:
      label=res[i['id']]
      dics[i['id']]=i['paragraphs'][label]
    f.write(json.dumps(dics))
  QA_test=QAdataset(tokenizer=tokenizer,split='test',data_dir='./',pre_label='./pre_label.json',data_path=args.test_path)
  batch_size=1
  QA_test_loader = DataLoader(
          QA_test,
          collate_fn=QA_test.collate_fn,
          shuffle=False,
          batch_size=batch_size,
      )
  
  QA_model=QuestionAnsweringModel(config,model_name,if_ckpt=True).to(device)
  ckpt = torch.load('./QA.pt')
  QA_model.load_state_dict(ckpt['model_state_dict'])

  QA_model.eval()
  with open(args.output,'w') as f:
    f.write('id,answer\n')
    for idx, batch in enumerate(tqdm(QA_test_loader)):
        inputs = batch
        example_id=inputs['example_id'][0]
        input_ids = inputs['input_ids'].to(device)
        offset_mapping = inputs["offset_mapping"]
        start_logits,end_logits = QA_model(
            input_ids=input_ids
        )

        best_prob=-1
        best_start=0
        best_end=0
        for i in range(len(start_logits)):
          for j in range(i,len(end_logits)):
            start_logit=start_logits[i].argmax(dim=-1)
            end_logit=end_logits[i].argmax(dim=-1)
            if i==j and start_logit>end_logit:
              pass
            else:
              if int(start_logit)*int(end_logit)==0:
                pass
              else:
                s_prob=start_logits[i][int(start_logit)]
                e_prob=end_logits[i][int(end_logit)]
                if float(s_prob+e_prob)>best_prob :
                  try:
                    start_start,start_end=offset_mapping[i][start_logit]
                    end_start,end_end=offset_mapping[j][end_logit]
                  except:
                    start_start,start_end=0,0
                    end_start,end_end=0,0
                  if (end_end-start_start)>30:
                    pass

                  else:
                    best_prob=float(s_prob+e_prob)
                    best_start=start_start
                    best_end=end_end
        context=inputs['context'][0]


        answer=context[best_start:best_end]
        if "「" in answer and "」" not in answer:
            answer += "」"
        elif "「" not in answer and "」" in answer:
            answer = "「" + answer
        if "《" in answer and "》" not in answer:
            answer += "》"
        elif "《" not in answer and "》" in answer:
            answer = "《" + answer
        answer = answer.replace(",", "")
        
        f.write(f'{example_id},{answer}\n')
        
def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--context_path", type=Path, default="./context.json")
    parser.add_argument("--test_path", type=Path, default="./test.json")
    parser.add_argument("--mc_ckpt", type=Path, default="./mul.pt")

    parser.add_argument("--qa_ckpt", type=Path, default="./QA.pt")
    parser.add_argument("--output", type=str, default="submission.csv")


    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()
    main(args)