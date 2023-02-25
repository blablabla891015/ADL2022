import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab
from torch import nn as nn
from torch import optim
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
# import wandb
import matplotlib.pyplot as plt

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len,mode="train")
        for split, split_data in data.items()
    }
    device="cuda"
    train_dataloader=DataLoader(dataset=datasets['train'],batch_size=2,shuffle=True,collate_fn=datasets['train'].collate_fn)
    eval_dataloader=DataLoader(dataset=datasets['eval'],batch_size=2,shuffle=False,collate_fn=datasets['eval'].collate_fn)
    # wandb.init()
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    num_class=len(tag2idx)+1
    model = SeqTagger(embeddings= embeddings,hidden_size=128,num_layers=2,dropout=0.4,bidirectional=False,num_class=num_class).to(device)
    # wandb.watch(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_acc_joint=0
    best_acc_token=0
    x=[]
    y=[]
    for epoch in epoch_pbar:
        model.train()
        
        joint_count=0
        joint_total=0
        token_count=0
        token_total=0
        for data in train_dataloader:
            output=model(data[0])
            # print(output)
            # print(data[1])
            loss=0
            for i,j in zip(output,data[1]):
                loss+=criterion(i,j)
                # if i.argmax(dim=-1)==j:
                #     joint_count+=1
                ans=True
                for pred_token,token in zip(i.argmax(dim=-1),j):
                    if pred_token==token:
                        token_count+=1
                    else:
                        ans=False
                    token_total+=1
                if ans:
                    joint_count+=1
                joint_total+=1
                
                
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(joint_count/joint_total)
        print(token_count/token_total)
        
        model.eval()
        joint_count_eval=0
        joint_total_eval=0
        token_count_eval=0
        token_total_eval=0
        for data in eval_dataloader:
            output=model(data[0])
            loss=0
            for i,j in zip(output,data[1]):
                loss+=criterion(i,j)
                ans=True
                for pred_token,token in zip(i.argmax(dim=-1),j):
                    if pred_token==token:
                        token_count_eval+=1
                    else:
                        ans=False
                    token_total_eval+=1
                if ans:
                    joint_count_eval+=1
                joint_total_eval+=1
        eval_acc_joint=joint_count_eval/joint_total_eval
        eval_acc_token=token_count_eval/token_total_eval
        print(eval_acc_joint)
        print(eval_acc_token)
        x.append(eval_acc_joint)
        y.append(int(epoch))
        if (eval_acc_joint)>best_acc_joint and (eval_acc_token)>best_acc_token:
            best_acc_joint=eval_acc_joint
            best_acc_token=eval_acc_token
            torch.save({'model_state_dict':model.state_dict()},'../ADL21-HW1/cache/slot/slot.pt')
            print('best model saved')
    plt.plot(y,x)
    plt.savefig('LSTM_100_slot.png')
    plt.show()
    # ckpt = torch.load('./cache/slot/slot.pt')
    # model.load_state_dict(ckpt['model_state_dict'])
    # model.to(device)
    # model.eval()
    # joint_count_eval=0
    # joint_total_eval=0
    # token_count_eval=0
    # token_total_eval=0
    # y_pred=[]
    # y_true=[]
    # for data in eval_dataloader:
    #         output=model(data[0])
    #         loss=0
    #         for i,j in zip(output,data[1]):
    #             y_pred.append(i.argmax(-1).tolist())
    #             y_true.append(j.tolist())
    #             loss+=criterion(i,j)
    #             ans=True
    #             for pred_token,token in zip(i.argmax(dim=-1),j):
    #                 if pred_token==token:
    #                     token_count_eval+=1
    #                 else:
    #                     ans=False
    #                 token_total_eval+=1
    #             if ans:
    #                 joint_count_eval+=1
    #             joint_total_eval+=1
    
        # eval_acc_joint=joint_count_eval/joint_total_eval
        # eval_acc_token=token_count_eval/token_total_eval
        # print(eval_acc_joint)
        # print(eval_acc_token)
    # id2tag=dict()
    # for key in tag2idx.keys():
    #     key
    #     x=int(tag2idx[key])
    #     id2tag[x]=key
    # x=[]
    # y=[]
    # for i in range(len(y_true)):
    #     res=[]
    #     res2=[]
    #     for j in range(len(y_true[0])):
    #         if y_true[i][j]!=9:
    #             res.append(id2tag[y_true[i][j]])
    #         if y_pred[i][j]!=9:
    #             res2.append(id2tag[y_pred[i][j]])
    #     x.append(res)
    #     y.append(res2)
    # print(x)
    # print(y)
            
    # print(classification_report(y, x, mode='strict', scheme=IOB2))
            
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=40)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)