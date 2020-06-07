from sklearn.metrics import confusion_matrix
from scipy.stats import hmean
import numpy as np

def binary_classification_report(y_true, y_pred, text=True):
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    size = len(y_true)
    
    accuracy = (tn + tp) / size
    random_accuracy = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / size ** 2
    kappa = (accuracy - random_accuracy) / (1 - random_accuracy)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = hmean([precision, recall])
    jaccard = tp / (tp + fn + fp)
    
    report = {
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'kappa': kappa,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard
    }
    
    if text:
        output = ''
        for metric, score in report.items():
            output += f'{metric.replace("_", " ")}: {score}\n'
        return output
    else:
        return report

def inspect_confusion_matrix(cmatrix):
    cmatrix = np.array(cmatrix)

    total_count = np.sum(cmatrix)
    tp = np.diag(cmatrix)
    fp = np.sum(cmatrix, axis=0) - tp
    fn = np.sum(cmatrix, axis=1) - tp
    tn = total_count - fp - fn - tp

    averaging = {
        'microavg': lambda score: score(*[np.mean(x) for x in (tp, fp, tn, fn)]),
        'macroavg': lambda score: np.mean(score(tp, fp, tn, fn))
    }

    results = {}

    results['noavg'] ={
        'total_count': total_count,
        'accuracy': tp.sum() / total_count
    }

    base_accuracy = (np.sum(cmatrix, axis=0) * np.sum(cmatrix, axis=1)).sum() / (total_count * total_count)
    results['noavg']['kappa'] = (results['noavg']['accuracy'] - base_accuracy) / (1 - base_accuracy)

    for averaging_name, averaging_f in averaging.items():
        scores = {
            'precision': averaging_f(lambda tp, fp, tn, fn: tp / (tp + fp)),
            'recall': averaging_f(lambda tp, fp, tn, fn: tp / (tp + fn)),
            'specificity': averaging_f(lambda tp, fp, tn, fn: tn / (tn + fp))
        }

        scores['sensitivity'] = scores['recall']
        try:
            scores['f1'] = hmean([scores['precision'], scores['recall']])
        except ValueError:
            scores['f1'] = None

        scores['ppv'] = scores['precision']
        scores['npv'] = averaging_f(lambda tp, fp, tn, fn: tn / (tn + fn)) 

        results[averaging_name] = scores

    return results