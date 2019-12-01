import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt
import scikitplot as skplt
import pickle
from scipy import stats
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

np.random.seed(0)

print("Loading dataset...")
df = pd.read_csv('data/train_baseline.csv', encoding='utf8', engine='python', chunksize=None)

# Define features and target
features = list(df.columns)
features.remove('label')

x_train, x_test, y_train, y_test = model_selection.train_test_split(df[features], df['label'], test_size=0.2, shuffle=True, stratify=df['label'])

print("Number of features:", len(features))
print("Features:", features)
print("")

# Standardize the feature data (mean 0, std 1)
standardize = lambda x: (x-x.mean()) / x.std()
for feature in features:
    df[feature] = df[feature].pipe(standardize)

# Remove outliers from training data
z = np.abs(stats.zscore(x_train))
x_train = x_train[(z < 3).all(axis=1)]
y_train = y_train[(z < 3).all(axis=1)]
print("Training set shape:", x_train.shape)

# k-nearest neighbors
knn = KNeighborsClassifier(5)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
knn_prob = knn.predict_proba(x_test)

# Random Forest
rf = ensemble.RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=100, bootstrap=True, class_weight=None, criterion="gini",max_depth=None, max_features="auto", max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, oob_score=False, verbose=0, warm_start=False)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
rf_prob = rf.predict_proba(x_test)

# SVM
svm = svm.SVC(gamma='scale', C=30000, probability=True)
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)
svm_prob = svm.predict_proba(x_test)

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnb_pred = gnb.predict(x_test)
gnb_prob = gnb.predict_proba(x_test)

# Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis()
qda = qda.fit(x_train, y_train)
qda_pred = qda.predict(x_test)
qda_prob = qda.predict_proba(x_test)

# Gaussian Process
#gp = GaussianProcessClassifier(1.0 * RBF(1.0))
#gp.fit(x_train, y_train)
#gp_pred = gp.predict(x_test)

# Evaluation
knn_fpr, knn_tpr, _ = metrics.roc_curve(y_test, knn_prob[:,1])
knn_auc = metrics.roc_auc_score(y_test, knn_prob[:,1])
rf_fpr, rf_tpr, _ = metrics.roc_curve(y_test, rf_prob[:,1])
rf_auc = metrics.roc_auc_score(y_test, rf_prob[:,1])
svm_fpr, svm_tpr, _ = metrics.roc_curve(y_test, svm_prob[:,1])
svm_auc = metrics.roc_auc_score(y_test, svm_prob[:,1])
gnb_fpr, gnb_tpr, _ = metrics.roc_curve(y_test, gnb_prob[:,1])
gnb_auc = metrics.roc_auc_score(y_test, gnb_prob[:,1])
qda_fpr, qda_tpr, _ = metrics.roc_curve(y_test, qda_prob[:,1])
qda_auc = metrics.roc_auc_score(y_test, qda_prob[:,1])


#svm_fpr, svm_tpr, _ = metrics.roc_curve(y_test, svm_pred)
#svm_auc = metrics.auc(svm_fpr, svm_tpr)

data = [knn_fpr, knn_tpr, knn_auc, rf_fpr, rf_tpr, rf_auc, svm_fpr, svm_tpr, svm_auc, gnb_fpr, gnb_tpr, gnb_auc, qda_fpr, qda_tpr, qda_auc]

with open('data/roc_baseline.data', 'wb') as filehandle:
    pickle.dump(data, filehandle)

plt.figure()
lw = 2
plt.plot(knn_fpr, knn_tpr, color='red', lw=lw, label='KNN (AUC = %0.2f)' % knn_auc)
plt.plot(rf_fpr, rf_tpr, color='blue', lw=lw, label='RF (AUC = %0.2f)' % rf_auc)
plt.plot(svm_fpr, svm_tpr, color='blue', lw=lw, label='ROC curve (area = %0.2f)' % svm_auc)
plt.plot(gnb_fpr, gnb_tpr, color='yellow', lw=lw, label='GNB (AUC = %0.2f)' % gnb_auc)
plt.plot(qda_fpr, qda_tpr, color='green', lw=lw, label='QDA (AUC = %0.2f)' % qda_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
