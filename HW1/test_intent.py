import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

from torch.utils.data import DataLoader
import torch.nn as nn


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len,mode='test')
    # TODO: crecate DataLoader for test dataset
    test_dataloader=DataLoader(dataset=dataset,batch_size=1,shuffle=False,collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        128,
        2,
        0.4,
        True,
        150,
    )
    device='cuda'
    model.to(device)
    
    model.eval()
    print(args.ckpt_path)
    ckpt = torch.load(args.ckpt_path)

    model.load_state_dict(ckpt['model_state_dict'])
    # load weights into model
    id2intent=dict()
    for key in intent2idx.keys():
        key
        x=int(intent2idx[key])
        id2intent[x]=key
    # TODO: predict dataset
    model.eval()
    pred_labels=[]
    ids=[]
    for data in test_dataloader:
        output=model(data[0])
        pred_label=int(output.argmax(dim=-1))
        id=data[1][0]
        pred_labels.append(id2intent[pred_label])
        ids.append(id)
    result_file = args.pred_file
    with open(result_file, 'w') as f:	
        f.write('id,intent\n')
        for i,j in zip(ids,pred_labels):
            f.write(f"{i},{j}\n")
        
        
        
    # TODO: write prediction to file (args.pred_file)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="intent_result.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
