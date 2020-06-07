import gc
from collections import Counter
from typing import Iterable, Union

import numpy as np
import torch
from bert_sequence_tagger.bert_utils import create_loader_from_flair_corpus
from libact.base.dataset import Dataset
from libact.base.interfaces import ProbabilisticModel
from libact.query_strategies import RandomSampling
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from al4ner.mc_dropout import DropoutMC, activate_mc_dropout


def find_in_between(offsets, start, end):
    res = []
    for i, offset in enumerate(offsets):
        if start <= offset and offset <= end:
            res.append(i)
    return res


def convert_y_to_bio_format(X, y):
    final_res = []
    for i, sent_y in enumerate(y):
        sent = X[i]
        offsets = []
        curr_offset = 0
        for index, word in enumerate(sent.split(' ')):
            offsets.append(curr_offset)
            curr_offset += len(word) + 1

        good_ys = ['O'] * len(sent)
        for w_y in sent_y:
            positions = find_in_between(offsets, w_y['start'], w_y['end'])
            good_ys[positions[0]] = 'B-' + w_y['tag']

            for pos in positions[1:]:
                good_ys[pos] = 'I-' + w_y['tag']

        final_res.append(good_ys)

    return final_res


class LibActNN(ProbabilisticModel):
    def __init__(self,
                 model_ctor,
                 trainer_ctor,
                 batch_size=16,
                 bs_pred=256,
                 retrain_epochs=3,
                 iter_retrain=1,
                 train_from_scratch=True,
                 valid_ratio=0.25,
                 string_input=True):
        self._model_ctor = model_ctor
        self._trainer_ctor = trainer_ctor
        self._model = None
        self._trainer = None
        self._batch_size = batch_size
        self._bs_pred = bs_pred
        self._retrain_epochs = retrain_epochs
        self._batch_size = batch_size
        self._iter_retrain = iter_retrain
        self._train_from_scratch = train_from_scratch
        self._valid_ratio = valid_ratio
        self._string_input = string_input

        self._iter = 0

    def _predict_core(self, X):
        if self._string_input:
            X = [sent.split(' ') for sent in X]

        torch.cuda.empty_cache()
        return self._model.predict(X)

    def predict_proba(self, X):
        return np.asarray(self._predict_core(X)[1]).reshape(-1, 1)

    def predict(self, X):
        return self._predict_core(X)[0]

    def train(self, libact_dataset, new_indexes=None):
        torch.cuda.empty_cache()
        def collate_fn(inpt): return tuple(zip(*inpt))

        if (new_indexes is not None) and (self._iter % self._iter_retrain) != 0:
            libact_dataset = Dataset([libact_dataset.data[i][0] for i in new_indexes],
                                     [libact_dataset.data[i][1] for i in new_indexes])
            n_epochs = 1
        else:
            n_epochs = self._retrain_epochs

        X, y = libact_dataset.format_sklearn()
        if self._string_input:
            y = convert_y_to_bio_format(X, y)
            X = [s.split(' ') for s in X]

        if self._valid_ratio > 0.:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self._valid_ratio)
            valid_data = list(zip(X_valid, y_valid))
        else:
            X_train, y_train = X, y
            valid_data = None

        train_data = list(zip(X_train, y_train))

        if (self._model is None) or self._train_from_scratch:
            self._model = self._model_ctor()
            self._trainer = self._trainer_ctor(self._model, len(X_train),
                                               train_data, valid_data)

            gc.collect()
            torch.cuda.empty_cache()

        self._trainer.train(self._retrain_epochs)

        self._iter += 1

    def score(self):
        pass


def find_most_common(row: Iterable[str], mode: Union['elem', 'count']):
    """
    Given iterable of words, return either most common element or its count
    """
    if mode == 'elem':
        return Counter(row).most_common(1)[0][0]
    elif mode == 'count':
        return Counter(row).most_common(1)[0][1]


class LibActNNBayes(LibActNN):
    def __init__(self,
                 model_ctor,
                 trainer_ctor,
                 batch_size=16,
                 bs_pred=256,
                 retrain_epochs=3,
                 iter_retrain=1,
                 train_from_scratch=True,
                 valid_ratio=0.25,
                 string_input=True,
                 n_estimators=10):
        super().__init__(model_ctor,
                         trainer_ctor,
                         batch_size,
                         bs_pred,
                         retrain_epochs,
                         iter_retrain,
                         train_from_scratch,
                         valid_ratio,
                         string_input)

        self._n_estimators = n_estimators

    def predict_proba(self, X):
        if self._string_input:
            X = [sent.split(' ') for sent in X]

        torch.cuda.empty_cache()
        self._model._bert_model.eval()
        activate_mc_dropout(self._model._bert_model, activate=True)

        ens_tags = []
        for i in range(self._n_estimators):
            tags, values = self._model.predict(X)
            tags = [np.array(item, dtype=object) for item in tags]
            ens_tags.append(tags)

        # sort metric proposed by Shen et al, 2018
        stacked_ans = [np.stack(item) for item in np.stack(ens_tags, -1)]
        # stacked_ans_result = [[find_most_common(row, 'elem') for row in ans.T] for ans in stacked_ans]
        stacked_ans_values = np.array([
            np.mean([find_most_common(row, 'count') / self._n_estimators for row in ans.T]) for ans in stacked_ans
        ])

        # deactivate mc dropout
        activate_mc_dropout(self._model._bert_model, activate=False)

        return stacked_ans_values.reshape(-1, 1)

    def _predict_core(self, X):

        # ensure dropout is deactivated for evaluation
        activate_mc_dropout(self._model._bert_model, activate=False)

        if self._string_input:
            X = [sent.split(' ') for sent in X]

        torch.cuda.empty_cache()
        return self._model.predict(X)


class LibActNNPositiveLessCertain(LibActNN):
    def __init__(self,
                 pos_label,
                 reduction_factor,
                 model_ctor,
                 trainer_ctor,
                 batch_size=16,
                 bs_pred=256,
                 retrain_epochs=3,
                 iter_retrain=1,
                 train_from_scratch=True,
                 valid_ratio=0.25):
        super().__init__(model_ctor,
                         trainer_ctor,
                         batch_size,
                         bs_pred,
                         retrain_epochs,
                         iter_retrain,
                         train_from_scratch,
                         valid_ratio)

        if isinstance(pos_label, Iterable):
            self._pos_label = set(pos_label)
        else:
            self._pos_label = set([pos_label])

        self._reduction_factor = reduction_factor

    def predict_proba(self, X):
        preds, probas = self._predict_core(X)
        for i in range(len(preds)):
            if any(tag in self._pos_label for tag in preds[i]):
                probas[i] *= self._reduction_factor

        return np.asarray(probas).reshape(-1, 1)


class RandomSamplingWithRetraining(RandomSampling):
    def __init__(self, *args, **kwargs):
        self._model = kwargs.pop('model', None)

        super().__init__(*args, **kwargs)

        self._model.train(self.dataset)

    def update(self, indexes, labels):
        self._model.train(self.dataset, indexes)

    def make_query(self):
        return super().make_query()
