import torch
import torch.nn as nn
from transformers import RobertaModel

from utils import word_lens_to_idxs_fast

class Base_Model(nn.Module):
    def __init__(self, config, task_name='ner'):
        super().__init__()
        self.config = config
        # encoder
        self.roberta_dim = 768 if 'base' in config.embedding_name else 1024
        self.roberta = RobertaModel.from_pretrained(config.embedding_name,
                                                    output_hidden_states=True)
        
        self.dropout = nn.Dropout(p=config.embedding_dropout)

    def encode(self, piece_idxs, attention_masks):
        batch_size, _ = piece_idxs.size()
        all_outputs = self.roberta(input_ids=piece_idxs, attention_mask=attention_masks)
        roberta_outputs = all_outputs[0]

        wordpiece_reprs = roberta_outputs[:, 1:-1, :]
        wordpiece_reprs = self.dropout(wordpiece_reprs)
        return wordpiece_reprs

    def encode_words(self, piece_idxs, attention_masks, word_lens):
        batch_size, _ = piece_idxs.size()
        all_outputs = self.roberta(input_ids=piece_idxs, attention_mask=attention_masks)
        roberta_outputs = all_outputs[0]
        cls_reprs = roberta_outputs[:, 0, :].unsqueeze(1)

        idxs, masks, token_num, token_len = word_lens_to_idxs_fast(word_lens)
        idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.roberta_dim) + 1
        masks = roberta_outputs.new(masks).unsqueeze(-1)
        roberta_outputs = torch.gather(roberta_outputs, 1,
                                    idxs) * masks
        roberta_outputs = roberta_outputs.view(batch_size, token_num, token_len, self.roberta_dim)
        roberta_outputs = roberta_outputs.sum(2)
        return roberta_outputs, cls_reprs

    def forward(self, batch):
        raise NotImplementedError

class _Embedding(Base_Model):
    def __init__(self, config, model_name='ner'):
        super(_Embedding, self).__init__(config, task_name=model_name)

    def get_tokenizer_inputs(self, batch):
        wordpiece_reprs = self.encode(
            piece_idxs=batch.piece_idxs,
            attention_masks=batch.attention_masks
        )
        return wordpiece_reprs

    def get_tagger_inputs(self, batch):
        word_reprs, cls_reprs = self.encode_words(
            piece_idxs=batch.piece_idxs,
            attention_masks=batch.attention_masks,
            word_lens=batch.word_lens
        )
        return word_reprs, cls_reprs