from vadim_ml.nlp import char_annotations_as_token_annotations, token_span_to_bio
from vadim_ml.io import load_file, dump_file
import random
import numpy as np
from seqeval.metrics import f1_score

def parse_ranges(text):
    for rng in text.split(','):
        boundaries = rng.split('-')
        start = int(boundaries[0])
        end = int(boundaries[-1])

        for x in range(start, end+1):
            yield x

from nltk.tokenize import TreebankWordTokenizer
tok = TreebankWordTokenizer()

def tokenize_row(row, annotation_cols):
    text = row['texts']
    token_spans = list(tok.span_tokenize(text))
    tokens = [text[s:e] for s, e in token_spans]

    text_dict = {
        'text': tokens
    }

    for pat in annotation_cols:
        spans = char_annotations_as_token_annotations(token_spans, row[pat])
        text_dict[pat] = token_span_to_bio(tokens, spans)

    return text_dict

def train_dev_test_distr(test_ratio, dev_ratio):
    x = random.random()
    if x < test_ratio:
        return 'test'
    elif x < test_ratio + dev_ratio:
        return 'dev'
    else:
        return 'train'

def entity_level_f1(model_folder):
    y_true = []
    y_pred = []

    for sentence in load_file(f'{model_folder}/test.tsv').split('\n\n'):
        true_bio_tags = []
        pred_bio_tags = []
        
        for token in sentence.split('\n'):
            if token:
                word, true, pred, prob = token.split(' ')

                true_bio_tags.append(true)
                pred_bio_tags.append(pred)                

        y_true.append(true_bio_tags)
        y_pred.append(pred_bio_tags)
 
    return f1_score(y_true, y_pred)


def print_al_stats(data_path, entire_dat_score):
    stats = np.load(data_path)
    print(stats[:,-1].shape)
    al_score = stats[:,-1].mean()
    score_ratio = al_score / entire_dat_score
    print('Al score: ', al_score)
    print('Entire data score:', entire_dat_score)
    print('Score ratio:', score_ratio)
