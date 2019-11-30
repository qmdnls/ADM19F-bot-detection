import numpy as np
import pandas as pd
import sklearn.ensemble as sk
import sklearn.metrics as metrics
import array as arr 
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

np.random.seed(0)

ints = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

print("Loading dataset...")
df = pd.read_csv('data/train.csv', encoding='utf8', engine='python', chunksize=None)

# Use 75% of entries as training data
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75
train, test = df[df['is_train']==True], df[df['is_train']==False]

features = list(df.columns)
features.remove('label')
y_train = train['label']


i=1
j=1
k=1
z=1
while i<28:
    ints[26]=ints[26]+1
    j=26;
    while j>0 : 
        if ints[j]>1:
            ints[j]=0
            ints[j-1]=ints[j-1]+1
        j=j-1
        
        #Main function should be here

	# Define features and target
        #features = list(df.columns)
        if ints[0]>0:
            features.remove('default_profile')
        if ints[1]>0:
            features.remove('default_profile_image')
        if ints[2]>0:
            features.remove('favourites_count')
        if ints[3]>0:
            features.remove('followers')
        if ints[4]>0:
            features.remove('following')
        if ints[5]>0:
            features.remove('listed_count')
        if ints[6]>0:
            features.remove('statuses_count')
        if ints[7]>0:
            features.remove('indegree_predecessors')
        if ints[8]>0:
            features.remove('indegree_successors')
        if ints[9]>0:
            features.remove('outdegree_predecessors')
        if ints[10]>0:
            features.remove('outdegree_successors')
        if ints[11]>0:
            features.remove('reputation_predecessors')
        if ints[12]>0:
            features.remove('reputation_successors')
        if ints[13]>0:
            features.remove('favorites_predecessors')
        if ints[14]>0:
            features.remove('favorites_successors')
        if ints[15]>0:
            features.remove('status_predecessors')
        if ints[16]>0:
            features.remove('status_successors')
        if ints[17]>0:
            features.remove('listed_predecessors')
        if ints[18]>0:
            features.remove('listed_successors')
        if ints[19]>0:
            features.remove('age_predecessors')
        if ints[20]>0:
            features.remove('age_successors')
        if ints[21]>0:
            features.remove('default_predecessors')
        if ints[22]>0:
            features.remove('defaults_successors')
        if ints[23]>0:
            features.remove('default_image_predecessors')
        if ints[24]>0:
            features.remove('default_image_successors')
        if ints[25]>0:
            features.remove('account_age')
        if ints[26]>0:
            features.remove('reputation')
            

        print("Number of features:", len(features))
        print("Features:", features)
	print("")

	# Random Forest
	rf = sk.RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=100, bootstrap=True,class_weight=None, criterion='gini',max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, oob_score=False, verbose=0, warm_start=False)
	rf.fit(train[features],y_train)
	rf_pred = rf.predict(test[features])

# Evaluation
	y_true = test['label']
	rf_fpr, rf_tpr, _ = metrics.roc_curve(y_true, rf_pred)
	rf_auc = metrics.auc(rf_fpr, rf_tpr)

	print("Evaluation")
	print("RF:", "TPR:", round(rf_tpr[1], 4), "FPR:", round(rf_fpr[1], 4), "F1 score:", round(metrics.f1_score(y_true, rf_pred), 4), "AUC:", round(rf_auc, 4))

    	k=0
    	z=1
    	while k<28:
        	if ints[k]==0:
        	    z=0
        	k=k+1
    	if z==1:
        	i=100

