import sklearn_crfsuite
from libact.base.dataset import Dataset
from libact.base.interfaces import ProbabilisticModel

import numpy as np


def word_features(sentence, i, use_chunks=False):
    # Get the current word and POS
    word = sentence[i][0]
    pos = sentence[i][1]
    
    features = { "bias": 1.0,
                 "word.lower()": word.lower(),
                 "word[-3:]": word[-3:],
                 "word[-2:]": word[-2:],
                 "word.isupper()": word.isupper(),
                 "word.istitle()": word.istitle(),
                 "word.isdigit()": word.isdigit(),
                 "pos": pos,
                 "pos[:2]": pos[:2],
               }
    # If chunks are being used, add the current chunk to the feature dictionary
    if use_chunks:
        chunk = sentence[i][2]
        features.update({ "chunk": chunk })
    # If this is not the first word in the sentence...
    if i > 0:
        # Get the sentence's previous word and POS
        prev_word = sentence[i-1][0]
        prev_pos = sentence[i-1][1]
        # Add characteristics of the sentence's previous word and POS to the feature dictionary
        features.update({ "-1:word.lower()": prev_word.lower(),
                          "-1:word.istitle()": prev_word.istitle(),
                          "-1:word.isupper()": prev_word.isupper(),
                          "-1:pos": prev_pos,
                          "-1:pos[:2]": prev_pos[:2],
                        })
        # If chunks are being used, add the previous chunk to the feature dictionary
        if use_chunks:
            prev_chunk = sentence[i-1][2]
            features.update({ "-1:chunk": prev_chunk })
    # Otherwise, add 'BOS' (beginning of sentence) to the feature dictionary
    else:
        features["BOS"] = True
    # If this is not the last word in the sentence...
    if i < len(sentence)-1:
        # Get the sentence's next word and POS
        next_word = sentence[i+1][0]
        next_pos = sentence[i+1][1]
        # Add characteristics of the sentence's previous next and POS to the feature dictionary
        features.update({ "+1:word.lower()": next_word.lower(),
                          "+1:word.istitle()": next_word.istitle(),
                          "+1:word.isupper()": next_word.isupper(),
                          "+1:pos": next_pos,
                          "+1:pos[:2]": next_pos[:2],
                        })
        # If chunks are being used, add the next chunk to the feature dictionary
        if use_chunks:
            next_chunk = sentence[i+1][2]
            features.update({ "+1:chunk": next_chunk })
    # Otherwise, add 'EOS' (end of sentence) to the feature dictionary
    else:
        features["EOS"] = True
    # Return the feature dictionary
    return features


def sentence_features(sentence, use_chunks=False):
    return [word_features(sentence, i, use_chunks) for i in range(len(sentence))]


# Return the label (NER tag) for each word in a given sentence
# def sentence_labels(sentence):
#     return [label for token, pos, chunk, label in sentence]


class LibActCrf(ProbabilisticModel):
    def __init__(self, *args, **kwargs):
        self.model = sklearn_crfsuite.CRF(*args, **kwargs)

    def train(self, libact_dataset, indexes=None):
        if indexes is not None:
            libact_dataset = Dataset([libact_dataset.data[i][0] for i in indexes], 
                                     [libact_dataset.data[i][1] for i in indexes])

        X, y = libact_dataset.format_sklearn()
        X = [sentence_features(sentence) for sentence in X]

        return self.model.fit(X, y)
    
    def fit(self, X, y):
        return self.model.fit([sentence_features(sentence) for sentence in X], y)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict([sentence_features(sent) for sent in feature], *args, **kwargs)

    def score(self, X, y, *args, **kwargs):
        #return self.model.score(X, y, **kwargs)
        pass

    def predict_proba(self, X, *args, **kwargs):
        all_labels = self.model.predict_marginals(X, *args, **kwargs)
        out = [np.mean([max(dict_.values()) for dict_ in l]) for l in all_labels]

        return np.array(out).reshape(-1, 1)
