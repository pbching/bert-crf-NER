import torch
from collections import namedtuple, Counter

instance_fields = [
    'sent_index', 'word_ids',
    'words', 'word_num',
    'piece_idxs', 'attention_masks', 'word_lens',
    'entity_label_idxs'
]
Instance = namedtuple('Instance', field_names=instance_fields)

train_instance_fields = [
    'words', 'word_num',
    'piece_idxs', 'attention_masks', 'word_lens',
    'entity_label_idxs'
]
Train_Instance = namedtuple('Train_Instance', field_names=train_instance_fields)

batch_fields = [
    'sent_index', 'word_ids',
    'words', 'word_num', 'word_mask',
    'piece_idxs', 'attention_masks', 'word_lens',
    'entity_label_idxs'
]
Batch = namedtuple('Batch', field_names=batch_fields)

train_batch_fields = [
    'words', 'word_num', 'word_mask',
    'piece_idxs', 'attention_masks', 'word_lens',
    'entity_label_idxs'
]
Train_Batch = namedtuple('Train_Batch', field_names=train_batch_fields)

def word_lens_to_idxs_fast(token_lens):
    max_token_num = max([len(x) for x in token_lens]) # max num of tokens
    max_token_len = max([max(x) for x in token_lens]) # max subword length
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len

def compute_word_reps_avg(piece_reprs, component_idxs):
    batch_word_reprs = []
    batch_size, _, _ = piece_reprs.shape
    _, num_words, _ = component_idxs.shape
    for bid in range(batch_size):
        word_reprs = []
        for wid in range(num_words):
            wrep = torch.mean(piece_reprs[bid][component_idxs[bid][wid][0]: component_idxs[bid][wid][1]], dim=0)
            word_reprs.append(wrep)
        word_reprs = torch.stack(word_reprs, dim=0)  # [num words, rep dim]
        batch_word_reprs.append(word_reprs)
    batch_word_reprs = torch.stack(batch_word_reprs, dim=0)  # [batch size, num words, rep dim]
    return batch_word_reprs

def word_lens_to_idxs(word_lens):
    max_token_num = max([len(x) for x in word_lens])
    max_token_len = max([max(x) for x in word_lens])
    idxs = []
    for seq_token_lens in word_lens:
        seq_idxs = []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.append([offset, offset + token_len])
            offset += token_len
        seq_idxs.extend([[-1, 0]] * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
    return idxs, max_token_num, max_token_len

def decode_from_bioes(tags):
    res = []
    ent_idxs = []
    cur_type = None

    def flush():
        if len(ent_idxs) > 0:
            res.append({
                'start': ent_idxs[0],
                'end': ent_idxs[-1],
                'type': cur_type})

    for idx, tag in enumerate(tags):
        if tag is None:
            tag = 'O'
        if tag == 'O':
            flush()
            ent_idxs = []
        elif tag.startswith('B-'):
            flush()
            ent_idxs = [idx]
            cur_type = tag[2:]
        elif tag.startswith('I-'):
            ent_idxs.append(idx)
            cur_type = tag[2:]
        elif tag.startswith('E-'):
            ent_idxs.append(idx)
            cur_type = tag[2:]
            flush()
            ent_idxs = []
        elif tag.startswith('S-'):
            flush()
            ent_idxs = [idx]
            cur_type = tag[2:]
            flush()
            ent_idxs = []

    flush()
    return res


def score_by_entity(pred_bioes_tag_sequences, gold_bioes_tag_sequences):
    assert (len(gold_bioes_tag_sequences) == len(pred_bioes_tag_sequences))

    def decode_all(tag_sequences):
        ents = []
        for sent_id, tags in enumerate(tag_sequences):
            for ent in decode_from_bioes(tags):
                ent['sent_id'] = sent_id
                ents += [ent]
        return ents

    gold_ents = decode_all(gold_bioes_tag_sequences)
    pred_ents = decode_all(pred_bioes_tag_sequences)

    correct_by_type = Counter()
    guessed_by_type = Counter()
    gold_by_type = Counter()

    for p in pred_ents:
        guessed_by_type[p['type']] += 1
        if p in gold_ents:
            correct_by_type[p['type']] += 1
    for g in gold_ents:
        gold_by_type[g['type']] += 1

    prec_micro = 0.0
    if sum(guessed_by_type.values()) > 0:
        prec_micro = sum(correct_by_type.values()) * 1.0 / sum(guessed_by_type.values())
    rec_micro = 0.0
    if sum(gold_by_type.values()) > 0:
        rec_micro = sum(correct_by_type.values()) * 1.0 / sum(gold_by_type.values())
    f_micro = 0.0
    if prec_micro + rec_micro > 0:
        f_micro = 2.0 * prec_micro * rec_micro / (prec_micro + rec_micro)

    return {
        'p': prec_micro * 100,
        'r': rec_micro * 100,
        'f1': f_micro * 100
    }