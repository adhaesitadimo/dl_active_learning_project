from sklearn.preprocessing import OneHotEncoder
from vadim_ml.pytorch import *

# We can overfit for as long as possible. Need a time limit
max_seconds_per_test = 10

def overfit_binary_clf():
    """Overfitting a primitive binary classifier"""

    clf = PytorchBinaryClassifier(nn.Sequential(
        nn.Linear(1, 10),
        nn.Sigmoid(),
        nn.Linear(10, 1)
    ), outputs='scores')
    clf.fit(np.array([[0.0], [1.0]], dtype=float), [0, 1], 
            np.array([[0.0], [1.0]], dtype=float), [0, 1])
    print(f'0 -> {clf.predict([0])}')
    print(f'1 -> {clf.predict([1])}')

def overfit_clf():
    """Overfiting a primitive 3-class classifier"""
    
    clf = PytorchClassifier(nn.Sequential(
        nn.Linear(3, 10),
        nn.Sigmoid(),
        nn.Linear(10, 10),
        nn.Sigmoid(),
        nn.Linear(10, 10),
        nn.Sigmoid(),
        nn.Linear(10, 3),
        nn.Softmax()
    ), outputs='probabilities')

    classification_X = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ]

    classification_y = ['art', 'science', 'bs']
    clf.schedule['min_epochs'] = 500
    clf.schedule['max_seconds'] = max_seconds_per_test

    clf.fit(classification_X, classification_y, classification_X, classification_y)
    print(clf.predict(classification_X))

def overfit_text_clf():
    """Overfitting CharConvMaxPool for text classification"""

    bible = 'Those things, which ye have both learned, and received, and heard, and seen in me, do: and the God of peace shall be with you.'
    not_bible = 'And it was like sooooooooooo like coooooool, duuude like'

    class ConvMaxPool(nn.Module):
        def __init__(self, embedding_size, class_count):
            super().__init__()
            self.convolutions = convolution_layers(embedding_size, class_count, (3, 3), conv_dim=1)

        def forward(self, x):
            x = x.transpose(-1, -2)
            x = self.convolutions(x)
            x = x.max(dim=-1)[0]
            return x

    ohenc = OneHotEncoder()

    X = np.array(ohenc.fit_transform([[char] for char in bible + not_bible]).todense())
    X = [X[:len(bible)], X[len(bible):]]
    y = ['bible', 'not bible']

    convpool = ConvMaxPool(len(ohenc.categories_[0]), 2)
    text_classifier = PytorchClassifier(convpool, batch=1, outputs='scores')
    text_classifier.schedule['max_seconds'] = max_seconds_per_test

    text_classifier.fit(X, y, X, y)
    print(f'bible -> {text_classifier.predict(X[:1])}')
    print(f'not bible -> {text_classifier.predict(X[1:])}')

if __name__ == '__main__':
    overfit_binary_clf()
    overfit_clf()
    overfit_text_clf()