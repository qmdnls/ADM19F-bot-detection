import numpy as np
import pandas as pd
import sklearn.ensemble as sk
import sklearn.metrics as metrics
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

np.random.seed(0)

print("Loading dataset...")
df = pd.read_csv('data/train.csv', encoding='utf8', engine='python', chunksize=None)

# Use 75% of entries as training data
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75
train, test = df[df['is_train']==True], df[df['is_train']==False]

# Define features and target
features = list(df.columns)
features.remove('label')
features.remove('is_train')
y_train = train['label']

print("Number of features:", len(features))
print("Features:", features)
print("")

# Random Forest
rf = sk.RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=100, bootstrap=True, class_weight=None, criterion='gini',max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, oob_score=False, verbose=0, warm_start=False)
rf.fit(train[features], y_train)
rf_pred = rf.predict(test[features])

# SVM
#svm = svm.SVC(kernel='linear')
#svm.fit(train[features], y_train)
#svm_pred = svm.predict(test[features])

# Evaluation
y_true = test['label']
rf_fpr, rf_tpr, _ = metrics.roc_curve(y_true, rf_pred)
rf_auc = metrics.auc(rf_fpr, rf_tpr)
#svm_fpr, svm_tpr, _ = metrics.roc_curve(y_true, svm_pred)
#svm_auc = metrics.auc(svm_fpr, svm_tpr)

print("Evaluation")
print("RF:", "TPR:", round(rf_tpr[1], 4), "FPR:", round(rf_fpr[1], 4), "F1 score:", round(metrics.f1_score(y_true, rf_pred), 4), "AUC:", round(rf_auc, 4))
#print("SVM:", "TPR:", round(svm_tpr[1], 4), "FPR:", round(svm_fpr[1], 4), "F1 score:", round(metrics.f1_score(y_true, svm_pred), 4), "AUC:", round(svm_auc, 4))

