import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
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

# Random Forest
rf = ensemble.RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=100, bootstrap=True, class_weight=None, criterion="gini",max_depth=None, max_features="auto", max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, oob_score=False, verbose=0, warm_start=False)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)

# SVM
svm = svm.SVC(gamma='scale', C=30000)
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnb_pred = gnb.predict(x_test)

# Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis()
qda = qda.fit(x_train, y_train)
qda_pred = qda.predict(x_test)

# Gaussian Process
gp = GaussianProcessClassifier(1.0 * RBF(1.0))
#gp.fit(x_train, y_train)
#gp_pred = gp.predict(x_test)

# Evaluation
knn_fpr, knn_tpr, _ = metrics.roc_curve(y_test, knn_pred)
knn_auc = metrics.auc(knn_fpr, knn_tpr)

rf_fpr, rf_tpr, _ = metrics.roc_curve(y_test, rf_pred)
rf_auc = metrics.auc(rf_fpr, rf_tpr)

svm_fpr, svm_tpr, _ = metrics.roc_curve(y_test, svm_pred)
svm_auc = metrics.auc(svm_fpr, svm_tpr)

gnb_fpr, gnb_tpr, _ = metrics.roc_curve(y_test, gnb_pred)
gnb_auc = metrics.auc(gnb_fpr, gnb_tpr)

qda_fpr, qda_tpr, _ = metrics.roc_curve(y_test, qda_pred)
qda_auc = metrics.auc(qda_fpr, qda_tpr)

print("Evaluation")
print("KNN:", "Acc:", round(metrics.accuracy_score(y_test, knn_pred), 4), "TPR:", round(knn_tpr[1], 4), "FPR:", round(knn_fpr[1], 4), "F1 score:", round(metrics.f1_score(y_test, knn_pred), 4), "AUC:", round(knn_auc, 4))
print("RF: ", "Acc:", round(metrics.accuracy_score(y_test, rf_pred), 4), "TPR:", round(rf_tpr[1], 4), "FPR:", round(rf_fpr[1], 4), "F1 score:", round(metrics.f1_score(y_test, rf_pred), 4), "AUC:", round(rf_auc, 4))
print("SVM:", "Acc:", round(metrics.accuracy_score(y_test, svm_pred), 4), "TPR:", round(svm_tpr[1], 4), "FPR:", round(svm_fpr[1], 4), "F1 score:", round(metrics.f1_score(y_test, svm_pred), 4), "AUC:", round(svm_auc, 4))
print("GNB:", "Acc:", round(metrics.accuracy_score(y_test, gnb_pred), 4), "TPR:", round(gnb_tpr[1], 4), "FPR:", round(gnb_fpr[1], 4), "F1 score:", round(metrics.f1_score(y_test, gnb_pred), 4), "AUC:", round(gnb_auc, 4))
print("QDA:", "Acc:", round(metrics.accuracy_score(y_test, qda_pred), 4), "TPR:", round(qda_tpr[1], 4), "FPR:", round(qda_fpr[1], 4), "F1 score:", round(metrics.f1_score(y_test, qda_pred), 4), "AUC:", round(qda_auc, 4))
