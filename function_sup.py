# coding=utf-8
# Functional functions that support PSONet running.

import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import io as sio
from sklearn import preprocessing
from fcmeans import FCM

def Apply_cva(T1, T2, no_sum=True):
    """
    CVA implementation.
    Args:
        T1: First temporal image with shape (nr, nc, ndim)
        T2: Second temporal image with shape (nr, nc, ndim)
        no_sum: If True, return CVA for each dimension separately
                If False, return summed CVA across all dimensions
    Returns:
        CVA result with shape:
            - (nr, nc, ndim) if no_sum=True
            - (nr, nc) if no_sum=False
    """
    if T1.shape != T2.shape:
        raise ValueError("T1 and T2 must have the same shape")
    nr, nc, ndim = T1.shape
    T1 = np.reshape(T1, newshape=[nr * nc, ndim])
    T2 = np.reshape(T2, newshape=[nr * nc, ndim])
    diff = np.asarray(T1 - T2, dtype=float)
    diff_s = (diff ** 2)
    if no_sum:
        cva_diff = np.sqrt(diff_s)
        cva_diff = np.reshape(cva_diff, newshape=[nr, nc, ndim])
    else:
        cva_diff = np.sqrt(np.sum(diff_s, axis=1))
        cva_diff = np.reshape(cva_diff, newshape=[nr, nc])
    return cva_diff

def Apply_accuracy_binary(pred, GT, index, fourshow=True):
    """
    Implementation of binary change detection accuracy evaluation
    Args:
        pred: Binary prediction (0=unchanged, 1=changed) with shape (nr, nc)
        GT: Binary ground truth (0=unchanged, 1=changed) with shape (nr, nc)
        index: Pixel index included in the precision calculation
        fourshow: If True, display TP, TN, FP, FN visualization
    Returns:
        Tuple of accuracy metrics: TP, TN, FP, FN, OA, TE, Kappa, F1, Recall, Precision, FA, MA
    """
    if pred.shape != GT.shape or pred.ndim != 2:
        print("You have made an error in your input! Please check and re-enter!")
        sys.exit()
    nr, nc = pred.shape
    pred = np.reshape(pred, newshape=[-1, nr * nc])
    pred = pred[:, index]
    GT[GT > 0] = 1
    GT = np.reshape(GT, newshape=[-1, nr * nc])
    GT = GT[:, index]
    TP = int(np.sum((pred == 1) & (GT == 1)))
    TN = int(np.sum((pred == 0) & (GT == 0)))
    FP = int(np.sum((pred == 1) & (GT == 0)))
    FN = int(np.sum((pred == 0) & (GT == 1)))
    OA = (TP + TN) / (TP + TN + FP + FN)
    Recall = (TP) / (TP + FN)
    Precision = (TP) / (TP + FP)
    F1 = 2 * ((Precision * Recall) / (Precision + Recall))
    FA = (FP) / (TN + FP)
    MA = (FN) / (TP + FN)
    TE = 1 - OA
    P = (((TP + FP) * (TP + FN)) / ((TP + TN + FP + FN) ** 2)) + (((FN + TN) * (FP + TN)) / ((TP + TN + FP + FN) ** 2))
    Kappa = (OA - P) / (1 - P)
    if fourshow:
        pred = np.reshape(pred, newshape=[nr, nc])
        GT = np.reshape(GT, newshape=[nr, nc])
        TP_loc = np.where((pred == 1) & (GT == 1))
        TN_loc = np.where((pred == 0) & (GT == 0))
        FP_loc = np.where((pred == 1) & (GT == 0))
        FN_loc = np.where((pred == 0) & (GT == 1))
        plt.axis('equal')
        area = 0.8 ** 2
        plt.scatter(TP_loc[1], nr - TP_loc[0], s=area, color='white', marker='.')  # TP-white
        plt.scatter(TN_loc[1], nr - TN_loc[0], s=area, color='black', marker='.')  # TN-black
        plt.scatter(FP_loc[1], nr - FP_loc[0], s=area, color='blue', marker='.')  # FP-blue
        plt.scatter(FN_loc[1], nr - FN_loc[0], s=area, color='yellow', marker='.')  # FN-yellow
        plt.savefig('fourshow.png', dpi=300)
        plt.show()
    print('Accuracy list:\n',
          'TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN, '\n',
          'OA:', '%.4f' % OA, 'TE:', '%.4f' % TE, 'Kappa:', '%.4f' % Kappa, 'F1:', '%.4f' % F1, '\n',
          'Recall:', '%.4f' % Recall, 'Precision:', '%.4f' % Precision,  'MA', '%.4f' % MA, 'FA', '%.4f' % FA
          )
    return TP, TN, FP, FN, OA, TE, Kappa, F1, Recall, Precision, FA, MA

def Apply_load_matdata(path, name, norm_flag=False, std_flag=False):
    """
    Implementation of matrix data (.mat) import operations
    Args:
        path: data path, string
        name: data name, string
        norm_flag: If True, apply Min-Max normalization
        std_flag: If True, apply Z-score standardization
    Returns:
        Loaded data with shape (nr, nc, ndim)
    """
    img = sio.loadmat(os.path.join(path, name))[name]
    print('Data_shape:', img.shape)
    if norm_flag:  # Max-Min normalization
        nr, nc, ndim = img.shape
        img = np.reshape(img, newshape=[nr * nc, ndim])
        img = preprocessing.MinMaxScaler().fit_transform(img)
        img = np.reshape(img, newshape=[nr, nc, ndim])
    if std_flag:  # z-score standardization
        nr, nc, ndim = img.shape
        img = np.reshape(img, newshape=[nr*nc, ndim])
        img = preprocessing.StandardScaler().fit_transform(img)
        img = np.reshape(img, newshape=[nr, nc, ndim])
    return img

def Apply_patches(X, win_size):
    """
    Extract patches from input data.
    Args:
        X: Input data with shape (nr, nc, ndim)
        win_size: patch size (win_size, win_size)
    Returns:
        Patches with shape (nr * nc, win_size, win_size, ndim)
    """
    margin = int((win_size) / 2)
    nr, nc, ndim = X.shape
    newX = np.zeros((nr + 2 * margin, nc + 2 * margin, ndim))
    newX[margin:nr + margin, margin:nc + margin, :] = X
    num = int(np.array(nr * nc))
    Patches = np.zeros((num, win_size, win_size, ndim)).astype(np.float16)
    for i in range(nr):
        for j in range(nc):
            idx = (nc*i) + j
            batch = newX[i:win_size+i, j:win_size+j, :]
            Patches[idx, :, :, :] = batch
    return Patches

def Apply_fcm(X, n_clusters=2, adjust=True):
    """
    Apply Fuzzy C-Means clustering
    Args:
        X: Input features with shape (nr, nc, ndim)
        n_clusters: Number of clusters
        adjust: If True, adjust labels based on class means
    Return:
        labels with shape (nr, nc)
        cluster centers
        membership values
    """
    if X.ndim == 2:
        X = np.reshape(X, newshape=[X.shape[0], X.shape[1], 1])
    nr, nc, ndim = X.shape
    X = np.reshape(X, newshape=[-1, X.shape[-1]])
    model_FCM = FCM(n_clusters=n_clusters)
    model_FCM.fit(X)
    affiliation = model_FCM.u
    X_label = model_FCM.u.argmax(axis=1)  # take out label value
    cluster_center = model_FCM.centers
    if adjust:
        abs_X = abs(X)
        diff_sum = np.zeros((2, n_clusters))
        for i in range(n_clusters):
            mean_value = (np.sum(abs_X[X_label == i])) / (np.sum(X_label == i))
            diff_sum[0, i] = mean_value
        diff_sum[1, :] = [k for k in range(n_clusters)]
        sort_diff_sum = diff_sum.T[np.lexsort(diff_sum[::-1, :])].T
        # adjust label and cluster_center
        cluster_center_adjust = np.zeros_like(cluster_center)
        affiliation_adjust = np.zeros_like(affiliation)
        for i in range(n_clusters):
            X_label[X_label == int(sort_diff_sum[1, i])] = i + n_clusters
            cluster_center_adjust[i, ] = cluster_center[int(sort_diff_sum[1, i]),]
            affiliation_adjust[:, i] = affiliation[:, int(sort_diff_sum[1, i])]
        X_label = X_label - n_clusters
        cluster_center = cluster_center_adjust  # Each line represents the center of clustering of a class,
        # e.g., the first line represents the center of clustering of class 0
        affiliation = affiliation_adjust
    X_label = np.reshape(X_label, newshape=[nr, nc])
    return X_label, cluster_center, affiliation