import numpy as np
import pandas as pd
import sklearn.ensemble as sk
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt
from sklearn import svm


np.random.seed(0)

print("Loading dataset...")
df = pd.read_csv('data/train.csv', encoding='utf8', engine='python', chunksize=None)

# Define features and target
features = list(df.columns)
features.remove('label')
features = ['favourites_count', 'followers', 'statuses_count', 'indegree_successors', 'outdegree_predecessors', 'favorites_predecessors', 'favorites_successors', 'status_predecessors', 'age_predecessors', 'account_age', 'listed_predecessors', 'reputation']

x_train, x_test, y_train, y_test = model_selection.train_test_split(df[features], df['label'], test_size=0.1, shuffle=True, stratify=df['label'])

print("Number of features:", len(features))
print("Features:", features)
print("")

# Random For3t
rf = sk.RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=100, bootstrap=True, class_weight=None, criterion='gini',max_depth=15, max_features=3, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, oob_score=False, verbose=0, warm_start=False)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)

# Evaluation
rf_fpr, rf_tpr, _ = metrics.roc_curve(y_test, rf_pred)
rf_auc = metrics.auc(rf_fpr, rf_tpr)

print("Evaluation")
print("RF:", "TPR:", round(rf_tpr[1], 4), "FPR:", round(rf_fpr[1], 4), "F1 score:", round(metrics.f1_score(y_test, rf_pred), 4), "AUC:", round(rf_auc, 4))

# Plot importance
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
