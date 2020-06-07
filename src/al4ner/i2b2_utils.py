import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch

import logging
logger = logging.getLogger('active_learning')


def drop_noise_samples(dataset, attr_name):
    # We select only positive examples and the whole document as a set of negative examples
    # if there are no annotations in the document at all
    keep = pd.Series([False for _ in range(dataset.shape[0])], index=dataset.index)
    for doc_id in dataset.doc_ids.unique():
        doc_dataset = dataset[dataset.doc_ids == doc_id]
        if doc_dataset[attr_name].astype(bool).sum() == 0:
            keep[doc_dataset.index] = True
    
    keep[dataset[dataset[attr_name].astype(bool)].index] = True
    return dataset[keep]



def split_train_test_by_document(dataset, test_ratio):
    doc_ids = dataset.doc_ids.unique()
    test = np.random.choice(list(range(dataset.shape[0])), size=int(dataset.shape[0] * test_ratio), replace =False)
    test_sample = set(test.tolist())
    train_sample = set(doc_ids.tolist()) - test_sample
    
    return dataset[dataset.doc_ids.isin(train_sample)], dataset[dataset.doc_ids.isin(test_sample)]


class SentenceGetter:
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

            
def upsample_good_tags(input_ids, tags, good_tag, ratio=1.):
    good_indexes = []
    for i in range(tags.shape[0]):
        if good_tag in tags[i]:
            good_indexes.append(i)
    
    chosen_indexes = np.random.choice(good_indexes, size=int(input_ids.shape[0] * ratio))
    return (np.concatenate((input_ids, input_ids[chosen_indexes]), axis=0), 
            np.concatenate((tags, tags[chosen_indexes]), axis=0))


def choose_only_good(input_ids, tags, masks, good_tag):
    good_indexes = []
    for i in range(tags.shape[0]):
        if good_tag in tags[i]:
            good_indexes.append(i)
    
    return input_ids[good_indexes], tags[good_indexes], masks[good_indexes]


def subsample_dataset(input_ids, tags, masks, positive_tag, negative_ratio, positive_ratio):
    negative_indexes = []
    positive_indexes = []
    for i in range(tags.shape[0]):
        if positive_tag in tags[i]:
            positive_indexes.append(i)
        else:
            negative_indexes.append(i)

    negative_indexes = np.random.choice(negative_indexes, size=int(len(negative_indexes) * negative_ratio))
    positive_indexes = np.random.choice(positive_indexes, size=int(len(positive_indexes) * positive_ratio))
    
    indexes = sorted(np.concatenate((negative_indexes, positive_indexes)))
    #positive_ids, positive_tags, positive_masks = choose_only_good(input_ids, tags, masks, good_tags)
    #negative_indexes = np.random.choice(negative_indexes, size=int(len(negative_indexes) * ratio))
    
    return input_ids[indexes], tags[indexes], masks[indexes]


def generate_logits_seqlogits_preds(seq_batch, masks, model):
    with torch.no_grad():
        logits = model(seq_batch, token_type_ids=None, attention_mask=masks).cpu().numpy()
    
    normalizer = masks.cpu().numpy().sum(axis=1)
    seq_logits = np.max(logits, axis=2).sum(axis=1) / normalizer
    preds = np.argmax(logits, axis=2)
    return logits, seq_logits, preds


def process_sentences(sent_batch, model, tokenizer, idx2tag):
    tokens = [tokenizer.tokenize(sent) for sent in sent_batch]
    seqs = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokens],
                         maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    masks = [[float(e > 0.) for e in seq] for seq in seqs]
    logits, seq_logits, preds = generate_logits_seqlogits_preds(torch.tensor(seqs), 
                                                                torch.tensor(masks), 
                                                                model)
    
    decoded = []
    real_logits = []
    for i in range(preds.shape[0]):
        decoded_seq = []
        real_logits_seq = []
        for j in range(preds.shape[1]):
            if masks[i][j]:
                decoded_seq.append(idx2tag[preds[i, j]])
                real_logits_seq.append(logits[i, j])
        
        decoded.append(decoded_seq)
        real_logits.append(real_logits_seq)
    
    return tokens, decoded, logits, seq_logits 


def has_attribute(doc_sents, attr_name):
    return (doc_sents[attr_name].apply(len) != 0).sum() != 0


def evaluation_level_document(model_results, dataset, attr_name):
    new_dataset = dataset.copy(True)
    new_dataset.index = range(dataset.shape[0])
    doc_ids = new_dataset.doc_ids.unique()
    
    tp = 0
    positive = 0
    pred_positive = 0
    for doc_id in doc_ids:
        doc_sents = new_dataset[new_dataset.doc_ids == doc_id]
        label = has_attribute(doc_sents, attr_name)
        
        pred_label = any('B' in e for e in np.array(model_results)[doc_sents.index])
        
        if label:
            positive += 1
            tp += int(label == pred_label)
        
        if pred_label:
            pred_positive += 1
    
    return positive, pred_positive, tp
