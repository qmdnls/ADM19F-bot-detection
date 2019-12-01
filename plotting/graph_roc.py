import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

np.random.seed(0)

# Load baseline
with open('../data/roc_baseline.data', 'rb') as filehandle:
    data = pickle.load(filehandle)
    knn_fpr, knn_tpr, knn_auc, rf_fpr, rf_tpr, rf_auc, svm_fpr, svm_tpr, svm_auc, gnb_fpr, gnb_tpr, gnb_auc, qda_fpr, qda_tpr, qda_auc = tuple(data)

# Load rf
with open('../data/roc_rf.data', 'rb') as filehandle:
    data = pickle.load(filehandle)
    rf_gf_fpr, rf_gf_tpr, rf_gf_auc = tuple(data)

# Load nn
with open('../data/roc_nn.data', 'rb') as filehandle:
    data = pickle.load(filehandle)
    nn_fpr, nn_tpr, nn_auc = tuple(data)

# Load nn_nf
with open('../data/roc_nn_nf.data', 'rb') as filehandle:
    data = pickle.load(filehandle)
    nn_nf_fpr, nn_nf_tpr, nn_nf_auc = tuple(data)

# Load nn_baseline
with open('../data/roc_nn_baseline.data', 'rb') as filehandle:
    data = pickle.load(filehandle)
    nn_baseline_fpr, nn_baseline_tpr, nn_baseline_auc = tuple(data)

plt.figure()
lw = 2
plt.plot(knn_fpr, knn_tpr, color='gold', lw=lw, label='KNN (AUC = %0.2f)' % knn_auc)
plt.plot(rf_fpr, rf_tpr, color='lightcoral', lw=lw, label='RF (AUC = %0.2f)' % rf_auc)
plt.plot(svm_fpr, svm_tpr, color='lightseagreen', lw=lw, label='SVM (area = %0.2f)' % svm_auc)
plt.plot(gnb_fpr, gnb_tpr, color='deeppink', lw=lw, label='GNB (AUC = %0.2f)' % gnb_auc)
plt.plot(qda_fpr, qda_tpr, color='black', lw=lw, label='QDA (AUC = %0.2f)' % qda_auc)
plt.plot(nn_fpr, nn_tpr, color='crimson', lw=lw, label='NN + NF + GF (AUC = %0.2f)' % nn_auc)
plt.plot(nn_nf_fpr, nn_nf_tpr, color='dodgerblue', lw=lw, label='NN + NF (AUC = %0.2f)' % nn_nf_auc)
plt.plot(nn_baseline_fpr, nn_baseline_tpr, color='deepskyblue', lw=lw, label='NN (AUC = %0.2f)' % nn_baseline_auc)
plt.plot(rf_gf_fpr, rf_gf_tpr, color='maroon', lw=lw, label='RF + NF + GF (AUC = %0.2f)' % rf_gf_auc)


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")

plt.show()
plt.savefig('../FIG/roc.pdf', dpi=300)
