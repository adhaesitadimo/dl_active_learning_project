from vadim_ml.metrics import *

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

class_count = 10
size = 500

class_distr = np.random.random(size=class_count)
class_distr /= class_distr.sum()

y_true = np.argmax(np.random.multinomial(1, class_distr, size=size), axis=1)
y_pred = np.argmax(np.random.multinomial(1, class_distr, size=size), axis=1)

print(classification_report(y_true, y_pred))
print(inspect_confusion_matrix(confusion_matrix(y_true, y_pred)))