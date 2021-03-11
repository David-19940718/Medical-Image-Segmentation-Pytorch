import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

__all__ = [
    "get_f1_score",
    "get_iou_score",
    "get_accuracy",
    "get_specificity",
    "get_sensitivity",
    "get_precision",
    "get_mae",
    "get_mse",
    "get_rmse",
    "get_score",
    "get_auc",
]


def get_f1_score(pd, gt, threshold=0.5):
    """
    :param threshold:
    :param pd: prediction
    :param gt: ground truth
    :return: dice coefficient or f1-score
    """

    pd = (pd > threshold).float()
    intersection = torch.sum((pd + gt) == 2)

    score = float(2 * intersection) / (float(torch.sum(pd) + torch.sum(gt)) + 1e-6)
    return score


def get_iou_score(pd, gt, threshold=0.5):
    """
    :param threshold:
    :param pd: prediction
    :param gt: ground truth
    :return: iou score or jaccard similarity
    """

    pd = (pd > threshold).float()
    intersection = torch.sum((pd + gt) == 2)
    union = torch.sum((pd + gt) >= 1)

    score = float(intersection) / (float(union) + 1e-6)
    return score


def get_accuracy(pd, gt, threshold=0.5):
    """
    formula = (tp + tn) / (tp + tn + fp + fn)
    :param threshold:
    :param pd: prediction
    :param gt: ground truth
    :return: accuracy score
    """

    pd = (pd > threshold).float()
    corr = torch.sum(pd == gt)
    tensor_size = pd.size(0) * pd.size(1) * pd.size(2) * pd.size(3)

    score = float(corr) / float(tensor_size)
    return score


def get_sensitivity(pd, gt, threshold=0.5):
    """
    formula = tp / (tp + fn)
    :param threshold:
    :param pd: prediction
    :param gt: ground truth
    :return: sensitivity or recall rate
    """

    pd = (pd > threshold).float()
    tp = (((pd == 1).float() + (gt == 1).float()) == 2).float()  # True Positive
    fn = (((pd == 0).float() + (gt == 1).float()) == 2).float()  # False Negative

    score = float(torch.sum(tp)) / (float(torch.sum(tp + fn)) + 1e-6)
    return score


def get_specificity(pd, gt, threshold=0.5):
    """
    formula = tn / (tn + fp)
    :param threshold:
    :param pd: prediction
    :param gt: ground truth
    :return: specificity score
    """
    pd = (pd > threshold).float()
    tn = (((pd == 0).float() + (gt == 0).float()) == 2).float()  # True Negative
    fp = (((pd == 1).float() + (gt == 0).float()) == 2).float()  # False Positive

    score = float(torch.sum(tn)) / (float(torch.sum(tn + fp)) + 1e-6)
    return score


def get_precision(pd, gt, threshold=0.5):
    """
    formula = tp / (tp + fn)
    :param threshold:
    :param pd: prediction
    :param gt: ground truth
    :return: precision score
    """
    pd = (pd > threshold).float()
    tp = (((pd == 1).float() + (gt == 1).float()) == 2).float()  # True Positive
    fp = (((pd == 1).float() + (gt == 0).float()) == 2).float()  # False Positive

    score = float(torch.sum(tp)) / (float(torch.sum(tp + fp)) + 1e-6)

    return score


def get_mae(pd, gt):
    """
    mean absolute error
    :param pd: prediction
    :param gt: ground truth
    :return: mae score
    """
    pd = torch.flatten(pd)
    gt = torch.flatten(gt)
    score = torch.mean(torch.abs(pd - gt))

    return score.item()


def get_mse(pd, gt):
    """
    mean squared error
    :param pd: prediction
    :param gt: ground truth
    :return: mse score
    """
    pd = torch.flatten(pd)
    gt = torch.flatten(gt)
    score = torch.mean((pd - gt) ** 2)

    return score.item()


def get_rmse(pd, gt):
    """
    root mean squared error
    :param pd: prediction
    :param gt: ground truth
    :return: rmse score
    """
    pd = torch.flatten(pd)
    gt = torch.flatten(gt)
    score = torch.sqrt(torch.mean((pd - gt) ** 2))

    return score.item()


def get_auc(pd, gt):
    fpr, tpr, _ = roc_curve(
        gt.flatten().cpu().detach().numpy().astype(np.uint8),
        pd.flatten().cpu().detach().numpy(),
        pos_label=1,
    )
    score = auc(fpr, tpr)

    return score


def get_score(pd, gt, mode='acc'):
    if mode == 'acc':
        return get_accuracy(pd, gt)
    elif mode == 'se':
        return get_sensitivity(pd, gt)
    elif mode == 'sp':
        return get_specificity(pd, gt)
    elif mode == 'pr':
        return get_precision(pd, gt)
    elif mode == 'iou':
        return get_iou_score(pd, gt)
    elif mode == 'dc':
        return get_f1_score(pd, gt)
    elif mode == 'mae':
        return get_mae(pd, gt)
    elif mode == 'mse':
        return get_mse(pd, gt)
    elif mode == 'rmse':
        return get_rmse(pd, gt)
    elif mode == 'auc':
        return get_auc(pd, gt)
    else:
        print('Please check the mode is available.')
        exit(0)
