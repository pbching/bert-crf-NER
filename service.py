import json
import torch
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from fastapi import FastAPI
from pydantic import BaseModel

from embedding import _Embedding
from utils import Batch, Instance, decode_from_bioes
from ner_classifier import NERClassifier, label2id, id2label
from config import config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wordpiece_splitter = RobertaTokenizer.from_pretrained(config.embedding_name)

class NERDatasetLive(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, batch):
        batch_sent_index = [inst.sent_index for inst in batch]
        batch_word_ids = [inst.word_ids for inst in batch]

        batch_words = [inst.words for inst in batch]
        batch_word_num = [inst.word_num for inst in batch]

        batch_piece_idxs = []
        batch_attention_masks = []
        batch_word_lens = []

        max_word_num = max(batch_word_num)
        max_wordpiece_num = max([len(inst.piece_idxs) for inst in batch])
        batch_word_mask = []
        batch_entity_label_idxs = []

        for inst in batch:
            batch_piece_idxs.append(inst.piece_idxs + [0] * (max_wordpiece_num - len(inst.piece_idxs)))
            batch_attention_masks.append(inst.attention_masks + [0] * (max_wordpiece_num - len(inst.piece_idxs)))
            batch_word_lens.append(inst.word_lens)
            batch_word_mask.append([1] * inst.word_num + [0] * (max_word_num - inst.word_num))
            batch_entity_label_idxs.append(inst.entity_label_idxs +
                                           [0] * (max_word_num - inst.word_num))

        batch_piece_idxs = torch.tensor(batch_piece_idxs, dtype=torch.int64).to(device)
        batch_attention_masks = torch.tensor(batch_attention_masks, dtype=torch.float).to(device)
        batch_entity_label_idxs = torch.tensor(batch_entity_label_idxs, dtype=torch.int64).to(device)
        batch_word_num = torch.tensor(batch_word_num, dtype=torch.int64).to(device)
        batch_word_mask = torch.tensor(batch_word_mask, dtype=torch.int64).eq(0).to(device)

        return Batch(
            sent_index=batch_sent_index,
            word_ids=batch_word_ids,
            words=batch_words,
            word_num=batch_word_num,
            word_mask=batch_word_mask,
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks,
            word_lens=batch_word_lens,
            entity_label_idxs=batch_entity_label_idxs
        )

def processed_inputs(sent_id, text):
    input_inst = []
    words = word_tokenize(text)
    word_ids = list(range(len(words)))
    pieces = [[p for p in wordpiece_splitter.tokenize(w) if p != '▁'] for w in words]
    for ps in pieces:
        if len(ps) == 0:
            ps += ['-']
    word_lens = [len(x) for x in pieces]
    flat_pieces = [p for ps in pieces for p in ps]
    if len(flat_pieces) > config.max_input_length - 2:
        cur_words = []
        cur_words_ids = []
        cur_flat_pieces = []
        for i in range(len(words)):
            if (len(cur_flat_pieces) + len(pieces[i])) > (config.max_input_length - 2):
                piece_idxs = wordpiece_splitter.encode(
                    cur_flat_pieces,
                    add_special_tokens=True,
                    max_length=config.max_input_length,
                    truncation=True
                )
                attn_masks = [1] * len(piece_idxs)
                piece_idxs = piece_idxs

                input_inst.append(Instance(
                    sent_index=sent_id,
                    word_ids=cur_words_ids,
                    words=cur_words,
                    word_num=len(cur_words),
                    piece_idxs=piece_idxs,
                    attention_masks=attn_masks,
                    word_lens=word_lens,
                    entity_label_idxs=[0 for _ in cur_words]
                ))

                cur_words = []
                cur_words_ids = []
                cur_flat_pieces = []

            cur_words.append(words[i])
            cur_words_ids.append(i)
            cur_flat_pieces.extend(pieces[i])
        
        if len(cur_flat_pieces) > 0:
            piece_idxs = wordpiece_splitter.encode(
                    cur_flat_pieces,
                    add_special_tokens=True,
                    max_length=config.max_input_length,
                    truncation=True
                )
            attn_masks = [1] * len(piece_idxs)
            piece_idxs = piece_idxs

            input_inst.append(Instance(
                sent_index=sent_id,
                word_ids=cur_words_ids,
                words=cur_words,
                word_num=len(cur_words),
                piece_idxs=piece_idxs,
                attention_masks=attn_masks,
                word_lens=word_lens,
                entity_label_idxs=[0 for _ in cur_words]
            ))

    else:
        piece_idxs = wordpiece_splitter.encode(
            flat_pieces,
            add_special_tokens=True,
            max_length=config.max_input_length,
            truncation=True
        )
        attn_masks = [1] * len(piece_idxs)
        piece_idxs = piece_idxs

        input_inst.append(Instance(
            sent_index=sent_id,
            word_ids=word_ids,
            words=words,
            word_num=len(words),
            piece_idxs=piece_idxs,
            attention_masks=attn_masks,
            word_lens=word_lens,
            entity_label_idxs=[0 for _ in words]
        ))

    return input_inst

# Load pretrained model
pretrained_weights = torch.load("path/to/checkpoint")
embedding = _Embedding(config)
embedding.to(device)
embedding_state_dict = embedding.state_dict()
for name, value in pretrained_weights.items():
    if name in embedding_state_dict:
        embedding_state_dict[name] = value
embedding.load_state_dict(embedding_state_dict)

ner_model = NERClassifier(config)
ner_model.to(device)
model_state_dict = ner_model.state_dict()
for name, value in pretrained_weights.items():
    if name in model_state_dict:
        model_state_dict[name] = value
ner_model.load_state_dict(model_state_dict)


class InputItem(BaseModel):
    doc: str=""

app = FastAPI()
# Infered API
@app.post('/api/ner')
def _ner_doc(item: InputItem):
    doc = item.doc
    sents = sent_tokenize(doc)
    data = []
    tag_results = []
    for id, sent in enumerate(sents):
        inputs_inst = processed_inputs(id, sent)
        tag_results.append({
            'id': id,
            'sentence': sent,
            'label': []
        })
        data.extend(inputs_inst)
    
    infer_dataset = NERDatasetLive(data)
    infer_dataloader = DataLoader(infer_dataset, batch_size=4, shuffle=False, collate_fn=infer_dataset.collate_fn)
    
    for batch in tqdm(infer_dataloader):
        word_reprs, cls_reprs = embedding.get_tagger_inputs(batch)
        pred_entity_labels = ner_model.predict(batch, word_reprs)
        # decode_from_bioes
        b_size = len(batch.word_num)
        for bid in range(b_size):
            sentid = batch.sent_index[bid]
            for i in range(batch.word_num[bid]):
                wordid = batch.word_ids[bid][i]
                text = batch.words[bid][i]
                tag = pred_entity_labels[bid][i]

                # NER tag
                tag_results[sentid]['label'].append({
                    'token': text,
                    'tag': tag
                })
    
    return {
        "tag_results": tag_results
    }

