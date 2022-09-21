import itertools
import numpy as np


def correlation_matrix(attribution_dic: dict[str, np.ndarray]) -> np.ndarray:
    """
    Computes the correlation matrix between the feature importance methods stored in a dictionary
    Args:
        attribution_dic: dictionary of the form feature_importance_method:feature_importance_scores

    Returns:
        Correlation matrix
    """
    corr_mat = np.empty((len(attribution_dic), len(attribution_dic)))
    for entry_id, (name1, name2) in enumerate(itertools.product(attribution_dic, attribution_dic)):
        corr_mat[entry_id//len(attribution_dic), entry_id%len(attribution_dic)] =\
            np.corrcoef(attribution_dic[name1].flatten(), attribution_dic[name2].flatten())[0, 1]
    return corr_mat


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count