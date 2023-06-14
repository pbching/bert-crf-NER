import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import RobertaModel, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ner_classifier import NERClassifier, label2id, id2label
from embedding import _Embedding
from utils import Train_Instance, Train_Batch, score_by_entity
from config import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("tner/ontonotes5")

trainset = dataset["train"]
trainset = trainset.map(lambda e: {'words': " ".join(e['tokens'])}, num_proc=4)
devset = dataset["test"]
devset = devset.map(lambda e: {'words': " ".join(e['tokens'])}, num_proc=4)

wordpiece_splitter = RobertaTokenizer.from_pretrained(config.embedding_name)


class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

    def numberize(self):
        data = []
        skip = 0
        for inst in self.data:
            words = inst['tokens']
            pieces = [[p for p in wordpiece_splitter.tokenize(w) if p != '▁'] for w in words]
            for ps in pieces:
                if len(ps) == 0:
                    ps += ['-']
            word_lens = [len(x) for x in pieces]
            assert 0 not in word_lens
            flat_pieces = [p for ps in pieces for p in ps]
            assert len(flat_pieces) > 0

            if len(flat_pieces) > config.max_input_length - 2:
                skip += 1
                continue
            piece_idxs = wordpiece_splitter.encode(
                flat_pieces,
                add_special_tokens=True,
                max_length=config.max_input_length,
                truncation=True
            )
            attn_masks = [1] * len(piece_idxs)
            assert len(piece_idxs) > 0

            entity_label_idxs = [tag for tag in inst['tags']]

            instance = Train_Instance(
                words=inst['tokens'],
                word_num=len(inst['tokens']),
                piece_idxs=piece_idxs,
                attention_masks=attn_masks,
                word_lens=word_lens,
                entity_label_idxs=entity_label_idxs
            )
            
            data.append(instance)
        print('Skipped {} over-length examples'.format(skip))
        print('Loaded {} examples'.format(len(data)))
        self.data = data

    def collate_fn(self, batch):
        batch_words = [inst.words for inst in batch]
        batch_word_num = [inst.word_num for inst in batch]

        batch_piece_idxs = []
        batch_attention_masks = []
        batch_word_lens = []
        batch_entity_label_idxs = []

        max_word_num = max(batch_word_num)
        max_wordpiece_num = max([len(inst.piece_idxs) for inst in batch])
        batch_word_mask = []

        for inst in batch:
            batch_piece_idxs.append(inst.piece_idxs + [0] * (max_wordpiece_num - len(inst.piece_idxs)))
            batch_attention_masks.append(inst.attention_masks + [0] * (max_wordpiece_num - len(inst.piece_idxs)))
            batch_word_lens.append(inst.word_lens)

            batch_entity_label_idxs.append(inst.entity_label_idxs +
                                           [0] * (max_word_num - inst.word_num))
            batch_word_mask.append([1] * inst.word_num + [0] * (max_word_num - inst.word_num))

        batch_piece_idxs = torch.tensor(batch_piece_idxs, dtype=torch.int64).to(device)
        batch_attention_masks = torch.tensor(batch_attention_masks, dtype=torch.float).to(device)
        batch_entity_label_idxs = torch.tensor(batch_entity_label_idxs, dtype=torch.int64).to(device)
        batch_word_num = torch.tensor(batch_word_num, dtype=torch.int64).to(device)
        batch_word_mask = torch.tensor(batch_word_mask, dtype=torch.int64).eq(0).to(device)

        return Train_Batch(
            words=batch_words,
            word_num=batch_word_num,
            word_mask=batch_word_mask,
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks,
            word_lens=batch_word_lens,
            entity_label_idxs=batch_entity_label_idxs
        )

train_set = NERDataset(trainset)
train_set.numberize()

dev_set = NERDataset(devset)
dev_set.numberize()


_embedding_layers = _Embedding(config)
_embedding_layers.roberta.encoder.gradient_checkpointing = True
_embedding_layers.to(device)

ner_model = NERClassifier(config)
ner_model.to(device)

model_parameters = [(n, p) for n, p in _embedding_layers.named_parameters()] + \
                                    [(n, p) for n, p in ner_model.named_parameters()]

param_groups = [
                {
                    'params': [p for n, p in model_parameters],
                    'lr': config.learning_rate, 'weight_decay': config.weight_decay
                }
            ]

optimizer = AdamW(params=param_groups)
batch_num = len(train_set) // config.batch_size
schedule = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=batch_num * config.num_train_epochs * 0.01,
                                        num_training_steps=batch_num * config.num_train_epochs)

train_dataloader = DataLoader(
            train_set, batch_size=config.batch_size,
            shuffle=True, collate_fn=train_set.collate_fn)

best_dev = {'p': 0, 'r': 0, 'f1': 0}
best_epoch = 0
eval_batch_num = len(dev_set) // config.batch_size
for epoch in range(1, config.num_train_epochs):
    print('*' * 30)
    print('NER: Epoch: {}'.format(epoch))
    # training
    _embedding_layers.train()
    ner_model.train()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(tqdm(train_dataloader)):
        word_reprs, cls_reprs = _embedding_layers.get_tagger_inputs(batch)
        loss = ner_model(batch, word_reprs)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            ner_model.parameters(), config.grad_clipping)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()
        print('NER: step: {}/{}, loss: {}'.format(batch_idx + 1, batch_num, loss.item()))
        
    ner_model.eval()
    # evaluate
    progress = tqdm(total=eval_batch_num, ncols=75,
                    desc='Eval epoch {}'.format(epoch))
    predictions = []
    golds = []
    for batch in DataLoader(dev_set, batch_size=config.batch_size,
                            shuffle=False, collate_fn=dev_set.collate_fn):
        progress.update(1)
        word_reprs, cls_reprs = _embedding_layers.get_tagger_inputs(batch)
        pred_entity_labels = ner_model.predict(batch, word_reprs)
        predictions += pred_entity_labels
        batch_entity_labels = batch.entity_label_idxs.data.detach().cpu().numpy().tolist()
        golds += [[id2label[l] for l in seq[:batch.word_num[i]]] for i, seq in enumerate(batch_entity_labels)]
    progress.close()
    dev_score = score_by_entity(predictions, golds)

    if dev_score['f1'] > best_dev['f1']:
        trainable_weight_names = [n for n, p in model_parameters if p.requires_grad]
        state = {}
        for k, v in _embedding_layers.state_dict().items():
            if k in trainable_weight_names:
                state[k] = v
        for k, v in ner_model.state_dict().items():
            if k in trainable_weight_names:
                state[k] = v
        ckpt_fpath=f'model_checkpoint/ckpt_{epoch}.pt'
        torch.save(state, ckpt_fpath)
        print('Saving weights to ... {})'.format(ckpt_fpath))

        best_dev = dev_score
        best_epoch = epoch

    # printout current best dev
    print('-' * 30)
    print('Best dev F1 score: epoch {}, F1: {:.2f}'.format(best_epoch, best_dev['f1']))

print('Training done!')