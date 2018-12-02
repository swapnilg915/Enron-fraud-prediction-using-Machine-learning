#!/usr/bin/python

import sys
import pickle
import json
sys.path.append("../tools/")

## import modules
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

## import preprocessing tool
from sklearn.preprocessing import MinMaxScaler
import numpy as np

## import validation tools
from sklearn.model_selection import train_test_split

## import classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier	

## import Evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

## import visualization tools
from matplotlib import pyplot as plt

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#======================================================================================================================

### Task 1: Create new feature(s)
for k,v in data_dict.items():
	# print k," = ",json.dumps(v, indent = 2)
	if data_dict[k]['from_this_person_to_poi'] != 'NaN' and data_dict[k]['from_poi_to_this_person'] != 'NaN':
		data_dict[k]['conversation_with_poi'] = data_dict[k]['from_this_person_to_poi'] + data_dict[k]['from_poi_to_this_person']
	else:
		data_dict[k]['conversation_with_poi'] = 'NaN'

	## removing outliers
	if data_dict[k]['exercised_stock_options'] != 'NaN' and data_dict[k]['to_messages'] != 'NaN':
		if data_dict[k]['exercised_stock_options'] > 300000000.0 and data_dict[k]['to_messages'] > 10000.0:
			print "\n outlier = ",data_dict[k]['exercised_stock_options'], data_dict[k]['to_messages']
			del data_dict[k]

### Store to my_dataset for easy export below.
my_dataset = data_dict


### Task 2: Select what features you'll use.

# features_list = ['poi','salary','to_messages','deferral_payments','total_payments','exercised_stock_options','shared_receipt_with_poi','total_stock_value','long_term_incentive','conversation_with_poi'] # You will need to use more features
# features_list = ['poi','exercised_stock_options']  

# this works best till now (without scaling)
# features_list = ['poi','exercised_stock_options','to_messages'] 

# # this works best till now (with scaling)
# features_list = ['poi','exercised_stock_options','to_messages','deferral_payments']

# # this works best till now (without scaling)
features_list = ['poi','exercised_stock_options','to_messages','conversation_with_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

############################################################ feature scalation

# scaler = MinMaxScaler()
# features = np.array(features)
# features = scaler.fit_transform(features)

### Task 3: plot with Removed outliers

# exercised_stock_options = []
# to_messages = []

# for index, data in enumerate(features):
# 	if data[0] < 300000000.0 and data[1] < 10000.0:
# 		exercised_stock_options.append(data[0])
# 		to_messages.append(data[1])
# 	if data[0] > 300000000.0 and data[1] > 10000.0:
# 		features.remove(data)

# print "\n max(to_messages) = ", max(to_messages)
# print "\n max(exercised_stock_options) = ",max(exercised_stock_options)
# plt.scatter( exercised_stock_options,to_messages)
# plt.xlabel("exercised_stock_options")
# plt.ylabel("to_messages")
# plt.show()

############################################################ dataset evaluation

print "\n len(data) = ",len(features), len(labels)
poi_num = 0
non_poi = 0
for label in labels:
	if label == 1.0:
		poi_num += 1
	else:
		non_poi += 1
print "\ntotal labels = ",len(labels),"\ntotal poi's = ",poi_num,"\nnon poi's = ",non_poi
print "\nNumber of POI in data (Percentage) = ",(poi_num / float(len(labels)))
print "\nNumber of NON-POI in data (Percentage) = ",(non_poi / float(len(labels)))

############################################################ cross validation

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

pca = PCA(svd_solver='randomized')
n_components = []
for i in range(1,len(features_list)):
		n_components.append(i)

clf_name = "DecisionTreeClassifier"

if clf_name == "DecisionTreeClassifier":
	clf2 = DecisionTreeClassifier()
	pipe = Pipeline(steps=[('PCA', pca), (clf_name, clf2)])
	min_samples_split = [2,4,5,7,10,15,17,20,25,30,40,50]
	estimator = GridSearchCV(pipe,dict(PCA__n_components= n_components,DecisionTreeClassifier__min_samples_split = min_samples_split))

elif clf_name == "SVC":	
	clf2 = SVC(kernel='rbf')
	pipe = Pipeline(steps=[('PCA', pca), (clf_name, clf2)])
	C = [1e3, 5e3, 1e4, 5e4, 1e5]
	gamma = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
	estimator = GridSearchCV(pipe,dict(PCA__n_components= n_components, SVC__C = C, SVC__gamma = gamma))

# print estimator.get_params().keys()

estimator.fit(features_train, labels_train)

print "\nBest estimator found by grid search on PCA and DT : \n",estimator.best_estimator_
print "\n estimator training accuracy = ",estimator.score(features_train, labels_train)
print "\n estimator testing acc = ",accuracy_score(estimator.predict(features_test), labels_test)
print "\n estimator confusion_matrix = \n",confusion_matrix(estimator.predict(features_test), labels_test)
print "\n estimator classification_report = \n",classification_report(estimator.predict(features_test), labels_test)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.

############################################################ classsifier 1 => GaussianNB
# clf = GaussianNB(priors=None)
# clf.fit(features_train, labels_train)

############################################################ classsifier 2 => DecisionTreeClassifier
# param_grid = {
# 		'criterion':['gini','entropy'],
#         'min_samples_split': [2,5,7,10,20,50],
#         }
# clf = GridSearchCV( DecisionTreeClassifier(), param_grid )

############################################################ classsifier 3 => KNeighborsClassifier

# clf = KNeighborsClassifier(3, p=1, weights = 'distance', n_jobs=-1)

# param_grid = {
#         'n_neighbors': [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40],
#         'p':[1,2],
#         }
# clf = GridSearchCV( KNeighborsClassifier(weights = 'distance'), param_grid )

############################################################ classsifier 4 => RandomForestClassifier
# param_grid = {
# 	'criterion':['entropy','gini'],
# 	'min_samples_split':[2,5,7,10,20,50],
# 	'n_estimators':[2,4,7,10,15,20,30,40,50,99],
# }
# clf = GridSearchCV(RandomForestClassifier(),param_grid)

############################################################ classsifier 5 => AdaBoostClassifier
param_grid = {
	'n_estimators':[2,4,7,10,15,20,30,40,50],
	'learning_rate':[0.01, 0.1,0.5,1.0]
}

clf = GridSearchCV(AdaBoostClassifier(),param_grid)

############################################################
# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

try:
	clf.fit(features_train, labels_train)

except Exception as e:
	print "=="*50
	print "\n Exception occured while fitting a classifier = ",e
	print "=="*50

print "\nBest classifier found by grid search: \n"
print clf.best_estimator_


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

## evaluation metric
print "\n training accuracy = ",clf.score(features_train, labels_train)
print "\n testing acc = ",accuracy_score(clf.predict(features_test), labels_test)
print "\n confusion_matrix = \n",confusion_matrix(clf.predict(features_test), labels_test)
print "\n classification_report = \n",classification_report(clf.predict(features_test), labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)