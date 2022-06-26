import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from src.models.encoder import TransformerInterEncoder
import argparse
from src.others.untilss import str2bool

parser = argparse.ArgumentParser()
parser.add_argument("-encoder", default='transformer', type=str,
                    choices=['transformer'])

parser.add_argument("-ff_size", default=2048, type=int)
parser.add_argument("-heads", default=8, type=int)
parser.add_argument("-inter_layers", default=2, type=int)
parser.add_argument("-dropout", default=0.1, type=float)

args = parser.parse_args()


class Bert(nn.Module):
    def __init__(self, bert_config):
        super(Bert, self).__init__()
        self.model = BertModel(bert_config)

    def forward(self, x, segs, mask):
        encoded_layers, _ = self.model(x, attention_mask=mask, token_type_ids=segs)
        # top_vec = encoded_layers[-1]
        return encoded_layers


class Summarizer(nn.Module):
    def __init__(self, device, bert_config_path=None):
        super(Summarizer, self).__init__()
        self.device = device
        self.bert_config = BertConfig.from_json_file(bert_config_path)
        self.bert = Bert(self.bert_config)
        self.encoder = TransformerInterEncoder(self.bert.model.config.hidden_size, args.ff_size, args.heads,
                                                   args.dropout, args.inter_layers)
        self.to(device)

    #def load_cp(self, pt):
       # self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):
        top_vec = self.bert(x, segs, mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls