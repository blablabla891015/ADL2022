from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn as nn

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.rnn=torch.nn.GRU(input_size=300,hidden_size=hidden_size,bidirectional=True,num_layers=num_layers,batch_first=True,dropout=dropout)
        # self.rnn=torch.nn.LSTM(input_size=300,hidden_size=hidden_size,bidirectional=True,num_layers=num_layers,batch_first=True,dropout=dropout)
        # self.rnn=torch.nn.RNN(input_size=300,hidden_size=hidden_size,bidirectional=True,num_layers=num_layers,batch_first=True,dropout=dropout)
        self.mlp=nn.Sequential(
            nn.LayerNorm(hidden_size*2),
            nn.Linear(hidden_size*2,hidden_size*2),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_size*2,num_class),
            nn.LeakyReLU()
        )
        # TODO: model architecture

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return 

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x=self.embed(batch)
        x,_=self.rnn(x)
        x= torch.mean(x, dim=1)
        x=self.mlp(x)
        return x


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x=self.embed(batch)
        x,_=self.rnn(x)
        #x= torch.mean(x, dim=1)
        x=self.mlp(x)
        return x
