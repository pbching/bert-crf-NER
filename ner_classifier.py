import torch
import torch.nn as nn

from CRF import CRFLoss, viterbi_decode

label2id = {
    "O": 0,
    "B-CARDINAL": 1,
    "B-DATE": 2,
    "I-DATE": 3,
    "B-PERSON": 4,
    "I-PERSON": 5,
    "B-NORP": 6,
    "B-GPE": 7,
    "I-GPE": 8,
    "B-LAW": 9,
    "I-LAW": 10,
    "B-ORG": 11,
    "I-ORG": 12, 
    "B-PERCENT": 13,
    "I-PERCENT": 14, 
    "B-ORDINAL": 15, 
    "B-MONEY": 16, 
    "I-MONEY": 17, 
    "B-WORK_OF_ART": 18, 
    "I-WORK_OF_ART": 19, 
    "B-FAC": 20, 
    "B-TIME": 21, 
    "I-CARDINAL": 22, 
    "B-LOC": 23, 
    "B-QUANTITY": 24, 
    "I-QUANTITY": 25, 
    "I-NORP": 26, 
    "I-LOC": 27, 
    "B-PRODUCT": 28, 
    "I-TIME": 29, 
    "B-EVENT": 30,
    "I-EVENT": 31,
    "I-FAC": 32,
    "B-LANGUAGE": 33,
    "I-PRODUCT": 34,
    "I-ORDINAL": 35,
    "I-LANGUAGE": 36
}
id2label = {v:k for k, v in label2id.items()}


class Linears(nn.Module):
    def __init__(self, dimensions, activation='relu', dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        return inputs

class NERClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.roberta_dim = 768 if 'base' in config.embedding_name else 1024
        self.entity_label_stoi = label2id
        self.entity_label_itos = id2label
        self.entity_label_num = len(self.entity_label_stoi)

        self.entity_label_ffn = Linears([self.roberta_dim, config.hidden_num,
                                         self.entity_label_num],
                                        dropout_prob=config.linear_dropout,
                                        bias=config.linear_bias,
                                        activation=config.linear_activation)

        self.crit = CRFLoss(self.entity_label_num)

    def forward(self, batch, word_reprs):
        batch_size, _, _ = word_reprs.size()
        logits = self.entity_label_ffn(word_reprs)
        loss, trans = self.crit(logits, batch.word_mask, batch.entity_label_idxs)
        return loss

    def predict(self, batch, word_reprs):
        batch_size, _, _ = word_reprs.size()

        logits = self.entity_label_ffn(word_reprs)
        _, trans = self.crit(logits, batch.word_mask, batch.entity_label_idxs)
        # decode
        trans = trans.data.cpu().numpy()
        scores = logits.data.cpu().numpy()
        bs = logits.size(0)
        tag_seqs = []
        for i in range(bs):
            tags, _ = viterbi_decode(scores[i, :batch.word_num[i]], trans)
            tags = [self.entity_label_itos[t] for t in tags]
            tag_seqs += [tags]
        return tag_seqs
    