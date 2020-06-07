from libact.base.interfaces import ProbabilisticModel
from libact.base.dataset import Dataset

import flair
from flair.models.sequence_tagger_model import START_TAG, STOP_TAG, argmax
from flair.data import Sentence, Corpus, Label
#from flair.trainers import ModelTrainer
from .trainer_flair import ModelTrainer as ModelTrainerFlair
from .mc_dropout import activate_mc_dropout, DropoutMC

import torch
import torch.nn.functional as F
import numpy as np
import os
import itertools


def make_flair_sentences(X, y=None, tag_type=None):
    sentences = [Sentence(' '.join(tokens)) for tokens in X]

    if y is not None:
        assert tag_type, 'Tag type is required if tags (y) are defined'

        for sentence, tags in zip(sentences, y):
            for (token, tag) in zip(sentence.tokens, tags):
                token.add_tag_label(tag_type, Label(tag))

    return sentences


def next_available_subfolder(path):
    for i in itertools.count():
        new_path = os.path.join(path, str(i))
        if os.path.exists(new_path):
            continue
        return new_path

    
def train_dev_split(sentences, dev_ratio=0.25):
    dev_size = len(sentences) * dev_ratio

    train = []
    dev = []

    for count, idx in enumerate(np.random.permutation(len(sentences))):
        if count < dev_size:
            dev.append(sentences[idx])
        else:
            train.append(sentences[idx])
        
    return Corpus(train=train, dev=dev, test=[])


def pad(seq_of_seqs, pad_with=0):
    padded_seqs = []
    max_len = max(len(seq) for seq in seq_of_seqs)

    for seq in seq_of_seqs:
        padded_seq = []

        for i in range(max_len):
            try:
                padded_seq.append(seq[i])
            except IndexError:
                padded_seq.append(pad_with)

        padded_seqs.append(padded_seq)
    
    return padded_seqs


class LibActFlair(ProbabilisticModel):
    """An adapter for training Flair SequenceTagger with libact.
       X is a corpus of tokenized sentences.
       y is a corpus of BIO sequences"""

    def __init__(self, tagger, base_path, mini_batch_size=32, eval_mini_batch_size=512,
                 reset_model_before_train=False, index_subset=True, save_all_models = False, **train_args):
        self._tagger = tagger
        self._index_subset = index_subset
        self._mini_batch_size = mini_batch_size
        self._eval_mini_batch_size = eval_mini_batch_size
        self._base_path = base_path
        self._train_args = train_args
        self._train_args['mini_batch_size'] = mini_batch_size
        self._save_all_models = save_all_models

        self._start_state = tagger.state_dict() if reset_model_before_train else None
        
    def _predict_batches(self, X):
        with torch.no_grad():
            if isinstance(X[0], str):
                # Got one sentence instead of a batch
                batches = [[X]]
            else:
                batches = [
                    X[x : x + self._mini_batch_size]
                    for x in range(0, len(X), self._mini_batch_size)
                ]

            batches = map(make_flair_sentences, batches)

            all_confidences = []
            all_tags = []

            for i, batch in enumerate(batches):
                with torch.no_grad():
                    self._tagger.predict(batch)
                    #self._tagger.predict(batch.copy()) # ???

                    for sentence in batch:
                        confidences, tags = [], []

                        for token in sentence.tokens:
                            tag = token.get_tag(self._tagger.tag_type)
                            confidences.append(tag.score)
                            tags.append(tag.value)

                        all_confidences.append(confidences)
                        all_tags.append(tags)

            return all_confidences, all_tags

    def predict_proba(self, X):
        confidences, tags = self._predict_batches(X)
        return np.array([np.mean(conf) for conf in confidences]).reshape(-1, 1)

    def predict(self, X):
        confidences, tags = self._predict_batches(X)
        return tags

    def score(self, X, y):
        sentences = make_flair_sentences(X, y, self._tagger.tag_type)
        result, _ = self._tagger.evaluate(sentences, eval_mini_batch_size=self._eval_mini_batch_size)
        return result.main_score

    def train(self, libact_dataset, indexes=None):
        if self._index_subset and indexes is not None:
            libact_dataset = Dataset([libact_dataset.data[i][0] for i in indexes], 
                                     [libact_dataset.data[i][1] for i in indexes])

        X, y = libact_dataset.format_sklearn()
        
        sentences = make_flair_sentences(X, y, self._tagger.tag_type)
        corpus = train_dev_split(sentences)

        if self._start_state:
            self._tagger.load_state_dict(self._start_state)

        if self._save_all_models:
            train_path = next_available_subfolder(self._base_path)
        else:
            train_path = self._base_path
        model_trainer = ModelTrainerFlair(self._tagger, corpus)
        return model_trainer.train(base_path=train_path, **self._train_args)

    
class LibActFlairBayes(LibActFlair):
    """An adapter for training Flair SequenceTagger with libact using Monte Carlo dropout.
       X is a corpus of tokenized sentences.
       y is a corpus of BIO sequences"""
    def __init__(self, tagger, base_path, mini_batch_size=32, eval_mini_batch_size=512,
                 reset_model_before_train=False, index_subset=True,
                 n_estimators=10, save_all_models = False, **train_args):
        self._tagger = tagger
        self._index_subset = index_subset
        self._mini_batch_size = mini_batch_size
        self._eval_mini_batch_size = eval_mini_batch_size
        self._base_path = base_path
        self._train_args = train_args
        self._train_args['mini_batch_size'] = mini_batch_size
        self._save_all_models = save_all_models

        self._start_state = tagger.state_dict() if reset_model_before_train else None
        self._n_estimators = n_estimators

    def predict_proba(self, X):
        print('PREDICT PROBA FOR BAYES')
        torch.cuda.empty_cache()
        self._tagger.eval()
        print("ACTIVATING MC DROPOUT")
        activate_mc_dropout(self._tagger, activate=True, verbose=True, option='flair')

        ens_tags = []
        for i in range(self._n_estimators):
            tags, values = self._predict_batches(X)
            tags = [np.array(item, dtype=object) for item in tags]
            ens_tags.append(tags)

        # sort metric proposed by Shen et al, 2018
        stacked_ans = [np.stack(item) for item in np.stack(ens_tags, -1)]
        # stacked_ans_result = [[find_most_common(row, 'elem') for row in ans.T] for ans in stacked_ans]
        stacked_ans_values = np.array([
            np.mean([find_most_common(row, 'count') / self._n_estimators for row in ans.T]) for ans in stacked_ans
        ])

        # deactivate mc dropout
        print("DEACTIVATING MC DROPOUT")
        activate_mc_dropout(self._tagger, activate=False, verbose=True, option='flair')

        return stacked_ans_values.reshape(-1, 1)

    def predict(self, X):
        print('PREDICT FOR BAYES')
        print("DEACTIVATING MC DROPOUT")
        activate_mc_dropout(self._tagger, activate=False, verbose=True, if_custom_rate=False, option='flair')
        confidences, tags = self._predict_batches(X)
        print("ACTIVATING MC DROPOUT")
        activate_mc_dropout(self._tagger, activate=True, verbose=True, if_custom_rate=False, option='flair')
        return tags
    
    
class PositiveLessCertain(ProbabilisticModel):
    def __init__(self, model):
        self._model = model

    def predict_proba(self, X):
        confidences, tags = self._model._predict_batches(X)
        proba = [ [confidence if tag[0] == 'O' else confidence / 2
                   for confidence, tag in zip(sentence_confidences, sentence_tags)]
                 for sentence_confidences, sentence_tags in zip(confidences, tags)]
        return np.array([np.mean(p) for p in proba]).reshape(-1, 1)

    def predict(self, X):
        return self._model.predict(X)

    def score(self, X, y):
        return self._model.score(X, y)

    def train(self, libact_dataset, indexes=None):
        print(f'indexes {indexes}')
        return self._model.train(libact_dataset, indexes)
