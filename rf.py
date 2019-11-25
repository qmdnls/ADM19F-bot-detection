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
features.remove('reputation_predecessors')
features.remove('reputation_successors')
features.remove('default_image_successors')
features.remove('default_image_predecessors')
features.remove('listed_successors')
features.remove('default_successors')
features.remove('default_predecessors')
features.remove('default_profile_image')

x_train, x_test, y_train, y_test = model_selection.train_test_split(df[features], df['label'], test_size=0.25, shuffle=True, stratify=df['label'])

print("Number of features:", len(features))
print("Features:", features)
print("")

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}



# Random Forest
rf = sk.RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=200, bootstrap=True, class_weight=None, criterion='gini',max_depth=15, max_features=19, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, oob_score=False, verbose=0, warm_start=False)
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
