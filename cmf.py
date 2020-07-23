import numpy as np

def accuracy(cm):
    tn, fp, fn, tp = cm.ravel()
    return (tp + tn) / cm.sum() if cm.sum() > 0 else 0

def precision(cm):
    tn, fp, fn, tp = cm.ravel()
    return (tp) / (tp + fp) if (tp + fp) > 0 else 0

def recall(cm):
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def phi(cm):
    a, b, c, d = cm.ravel()
    return (a*d - b*c) / np.sqrt((a+b)*(c+d)*(a+c)*(b+d))

def positive(cm):
    tn, fp, fn, tp = cm.ravel()
    return (tp + fn) 
