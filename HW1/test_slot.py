import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    data_path=args.data_dir
    data = json.loads(data_path.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len,mode='test')
    # TODO: crecate DataLoader for test dataset
    test_dataloader=DataLoader(dataset=dataset,batch_size=1,shuffle=False,collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        embeddings,
        128,
        2,
        0.4,
        True,
        10,
    )
    id2tag=dict()
    for key in tag2idx.keys():
        key
        x=int(tag2idx[key])
        id2tag[x]=key
    device='cuda'
    print(args.ckpt_dir)
    ckpt = torch.load(args.ckpt_dir)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    results=[]
    ids=[]
    for data in test_dataloader:
        output=model(data[0])
        length=int(data[1][0])
        id=data[2][0]
        for i in output:
            res=i.argmax(dim=-1)
            ans=[]
            for j in res[:length]:
                ans.append(id2tag[int(j)])
            results.append(ans)
        ids.append(id)
    result_file=args.pred_file
    with open(result_file, 'w') as f:
        f.write("id,tags\n")
        for id,tags in zip(ids,results):
            tags=" ".join(tags)
            f.write(f"{id},{tags}\n")
    
    
        
                
                
    
    


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
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

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