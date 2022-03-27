import dgl
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import pydot
import os
import collections
import random
from model import HAN
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
import torch
import json
from GraphGenerator import graphGenerator_ethersolve
from GraphGenerator import graphGenerator_slither
from utils import readfile
from utils import evaluate
from utils import model_select
from utils import fullset
from DataLoader import dataloader

time1_y_true = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
time1_source = np.array([1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1])
time1_runtim = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
time1_creati = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1])
def dis(y_true,y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
    return fpr, tpr,display
a = []
fpr1, tpr1,display1 = dis(time1_y_true,time1_source)
fpr2, tpr2,display2 = dis(time1_y_true,time1_runtim)
fpr3, tpr3,display3 = dis(time1_y_true,time1_creati)
plt.plot(fpr1, tpr1,'g','sourcecode')
plt.plot(fpr2, tpr2,'b','runtime')
plt.plot(fpr3, tpr3,'r','creationcode')
plt.show()