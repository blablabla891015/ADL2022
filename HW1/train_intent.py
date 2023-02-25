import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from model import  SeqClassifier
import torch
from tqdm import trange
from dataset import SeqClsDataset
from utils import Vocab
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
import matplotlib.pyplot as plt

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len,mode="train")
        for split, split_data in data.items()
    }
    device="cuda"
    # wandb.init()
    # TODO: crecate DataLoader for train / dev datasets
    train_dataloader=DataLoader(dataset=datasets['train'],batch_size=2,shuffle=True,collate_fn=datasets['train'].collate_fn)
    eval_dataloader=DataLoader(dataset=datasets['eval'],batch_size=2,shuffle=False,collate_fn=datasets['eval'].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    num_class=len(intent2idx)
    model = SeqClassifier(embeddings= embeddings,hidden_size=128,num_layers=2,dropout=0.4,bidirectional=True,num_class= num_class).to(device)
    # wandb.watch(model)
    # TODO: init optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    # optimizer = optim.Adam(model.parameters(), lr=2e-5)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_acc=0
    epoch_axis=[]
    eval_axis=[]
    for epoch in epoch_pbar:
        model.train()
        total_total=0
        total_count=0
        for data in train_dataloader:
            
            output=model(data[0])
            label=data[1]
            loss=criterion(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count=0
            total=0
            for i,j in zip(output.argmax(dim=1),label):
                if int(i)==int(j):
                    count+=1
                total+=1
            total_count+=count
            total_total+=total
        print(total_count/total_total)
        
        model.eval()
        total_total_eval=0
        total_count_eval=0
        for data in eval_dataloader:
            
            output=model(data[0])
            label=data[1]
            # loss=criterion(output,label)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            count=0
            total=0
            for i,j in zip(output.argmax(dim=1),label):
                if int(i)==int(j):
                    count+=1
                total+=1
            total_count_eval+=count
            total_total_eval+=total
        eval_acc=total_count_eval/total_total_eval
        print(total_count_eval/total_total_eval)
        epoch_axis.append(int(epoch))
        eval_axis.append(eval_acc)
        if(eval_acc>best_acc):
            best_acc=eval_acc
            torch.save({'model_state_dict':model.state_dict()},'../ADL21-HW1/cache/intent/intent.pt')
            print('best model saved')
    plt.plot(epoch_axis,eval_axis)
    plt.savefig('LSTM_100.png')
    plt.show()
        

            
            
                    
                
                
                    
                
            

            #output = model(data)
            # data=[i.to(device) for i in data]
            
            
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
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
