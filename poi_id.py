#!/usr/bin/python

import sys
import pickle
import pandas as pd
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt  
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest, f_classif
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn import tree, feature_selection
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn import grid_search
from sklearn.metrics import precision_score
from tester import test_classifier, dump_classifier_and_data



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list =['poi','exercised_stock_options', 'bonus', 'total_stock_value']
features_considered = ['poi', 'salary', 'deferral_payments', 
                      'total_payments', 'exercised_stock_options', 
                      'bonus', 'restricted_stock', 'total_stock_value', 
                      'expenses', 'loan_advances', 'other', 
                      'deferred_income', 'long_term_incentive', 
                      'to_messages', 'from_messages', 
                      'from_this_person_to_poi', 'from_poi_to_this_person', 
                      'shared_receipt_with_poi','communicationRatio', 
                      'nonSalaryRatio']
features1 = ['poi', 'exercised_stock_options', 'bonus', 'total_stock_value']
features2 = ['poi', 'salary', 'exercised_stock_options', 'bonus', 
			'total_stock_value', 'deferred_income'] 
features3 = ['poi', 'from_messages', 'from_this_person_to_poi', 
			'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
outlier_list = ['TOTAL', 'LOCKHART EUGENE E','THE TRAVEL AGENCY IN THE PARK']
for i in outlier_list:   
    data_dict.pop(str(i),0)
    print "Removed outlier ", i

### Correct wrong input
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093

### Task 3: Create new feature(s)
def create_new_features():
	"""Create four new features: 'nonSalary', 'communicationWithPoi',
	'nonSalaryRatio', 
	"""
	# Non-salary Income
	for person in data_dict.values():
	    nonSalary = 0
	    nonSalary_list = ['deferral_payments', 'total_payments', 
	    				  'exercised_stock_options', 'bonus',
	                      'restricted_stock', 'total_stock_value', 'expenses', 
	                      'loan_advances', 'other', 'deferred_income', 
	                      'long_term_incentive']
	    for i in nonSalary_list:
	        if person[i] != 'NaN':
	            nonSalary = nonSalary + person[i]
	    if nonSalary == 0:
	        nonSalary = 'NaN'
	    person['nonSalary'] = nonSalary

	# Total email communication with Poi
	for person in data_dict.values():
	    communicationWithPoi = 0
	    communicationWithPoi_list = ['from_this_person_to_poi',
	                                 'from_poi_to_this_person', 
	                                 'shared_receipt_with_poi']
	    for i in communicationWithPoi_list:
	        if person[i] != 'NaN':
	            communicationWithPoi = communicationWithPoi + person[i]
	    if communicationWithPoi == 0:
	        communicationWithPoi = 'NaN'
	    person['communicationWithPoi'] = communicationWithPoi

	# Ratios of non-salary income/total income, 
	# total communication with poi/total communication
	for person in data_dict.values():
	    if (person['nonSalary'] !='NaN') & (person['salary'] !='NaN'):
	        person['nonSalaryRatio'] = \
	        float(person['nonSalary'])/(person['nonSalary'] \
	        	+ person['salary'])
	    else:
	        person['nonSalaryRatio'] = 'NaN'
	    if (person['communicationWithPoi'] !='NaN') & (
	    	person['to_messages']!='NaN') & (person['from_messages'] !='NaN'):
	        person['communicationRatio'] = \
	        float(person['communicationWithPoi'])/(
	        	person['communicationWithPoi'] + \
	        	person['to_messages'] + person['from_messages'])
	    else:
	        person['communicationRatio'] = 'NaN'
	print "New Feature Created. "

create_new_features()


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_considered, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 4: Try a varity of classifiers
def al_tuning(features_list):
	"""Get the best param for classifier"""
	### Extract features and labels from dataset for local testing
	data = featureFormat(my_dataset, features_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)
	features_train, features_test, labels_train, labels_test = \
		train_test_split(features, labels, test_size=0.3, random_state=42)

	print features_list
	# Decision Tree
	parameters= {             
	              'min_samples_split': [2, 4, 6,8,10,20],
	              'criterion': ['entropy','gini']
	             }
	from sklearn.tree import DecisionTreeClassifier
	DT = DecisionTreeClassifier()
	clf = grid_search.GridSearchCV(estimator=DT, 
		param_grid=parameters,scoring='f1') 
	clf.fit(features_train, labels_train)
	print "Best parameters for Decision Tree classfier: ", clf.best_params_ 
	print "F Score for Decision Tree Classifier: ",clf.best_score_ 

	# Gaussian Naive Bayes
	parameters= {    
            }
	NB = GaussianNB()
	clf = grid_search.GridSearchCV(estimator=NB,param_grid=parameters,
		scoring='f1') 
	clf.fit(features_train, labels_train)
	print "Best parameters for Naive Bayes classfier: ", clf.best_params_ 
	print "F Score for Naive bayes Classifier: ",clf.best_score_  

	#RandomForestClassifier
	parameters= {             
                'n_estimators': [10,50,100],
                'criterion': ['entropy','gini'],
                'min_samples_split': [2, 4, 6,8,10]
                 }
	from sklearn.ensemble import RandomForestClassifier
	RF = RandomForestClassifier()
	clf = grid_search.GridSearchCV(estimator=RF,
			param_grid=parameters,scoring='f1') 
	clf.fit(features_train, labels_train)
	print "Best parameters for Random forest classfier: ", clf.best_params_ 
	print "F Score for Random Forest Classifier: ",clf.best_score_ 



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Chosen classifier
clf = tree.DecisionTreeClassifier(min_samples_split=8,criterion= 'entropy')
test_classifier(clf, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


### Utilities

def print_NaN():
	"""Print the percentage of NaN in each group"""
	print "|feature| poi |non-p | all  |  % | "
	print "|----|------|------|------|---------|  "
	for feature in allFeatures:
	    poiCount = 0
	    count = 0
	    for name in data_dict.keys():
	        if data_dict[name][feature] != "NaN":
	            count += 1
	            if data_dict[name]['poi']:
	                poiCount += 1
	    print '|', feature , 
	    print '|{0:.2f}'.format(poiCount/18.0), 
	    print '| {0:.2f}'.format((count - poiCount)/(len(data_dict)-18.0)),
	    print '| {0:.2f} |'.format(count/143.0),
	    print '{0:.2f} |'.format(float(poiCount)/count)

def omit_NaN(d_dict):
    """
    Remove 'NaN' value and allow pandas to assign the np.nan missing value
    """
    for name in d_dict.keys():
        d = dict([(field, d_dict[name][field]) \
        	for field in d_dict[name].keys() \
        	if d_dict[name][field] != "NaN"])
        d['name'] = name
        yield(d)
	df = pd.DataFrame(omit_NaN(data_dict))
	numeric_cols = [col for col in df.columns \
					if col not in ['name','email_address', 'poi']]
	for col in numeric_cols:
	    df[col] = df[col].astype('float32')


def plot_new_features():
	"""Plot the four new features"""
	# Plot the new features
	sb.plt.figure(figsize=(16,10))
	sb.plt.subplot(221)
	sb.stripplot(x="poi", y="nonSalary", data=df, jitter=True)
	sb.plt.subplot(222)
	sb.stripplot(x="poi", y="communicationWithPoi", data=df, jitter=True)
	sb.plt.subplot(223)
	sb.stripplot(x="poi", y="nonSalaryRatio", data=df, jitter=True)
	sb.plt.subplot(224)
	sb.stripplot(x="poi", y="communicationRatio", data=df, jitter=True)
	sb.plt.show() 


def get_kbest_score():
	"""Get feature score from SelectKBest feature selection"""
	transform = SelectKBest(f_classif)
	transform.fit(features, labels)
	for i in range(len(features_considered)-1):
	    print features_considered[i+1], ": ", transform.scores_[i]

def get_score(clf, dataset, feature_list, return_accuracy = False, 
	return_precision = False, return_recall = False, return_f1 = False, 
	folds = 1000):
	"""Utility function to get scores for valuation"""
	data = featureFormat(dataset, feature_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)
	cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
	true_negatives = 0
	false_negatives = 0
	true_positives = 0
	false_positives = 0
	for train_idx, test_idx in cv: 
	    features_train = []
	    features_test  = []
	    labels_train   = []
	    labels_test    = []
	    for ii in train_idx:
	        features_train.append( features[ii] )
	        labels_train.append( labels[ii] )
	    for jj in test_idx:
	        features_test.append( features[jj] )
	        labels_test.append( labels[jj] )
	    
	    ### fit the classifier using training set, and test on test set
	    clf.fit(features_train, labels_train)
	    predictions = clf.predict(features_test)
	    for prediction, truth in zip(predictions, labels_test):
	        if prediction == 0 and truth == 0:
	            true_negatives += 1
	        elif prediction == 0 and truth == 1:
	            false_negatives += 1
	        elif prediction == 1 and truth == 0:
	            false_positives += 1
	        elif prediction == 1 and truth == 1:
	            true_positives += 1
	        else:
	            print "Warning: Found a predicted label not == 0 or 1."
	            print "All predictions should take value 0 or 1."
	            print "Evaluating performance for processed predictions:"
	            break
	try:
	    total_predictions = true_negatives + false_negatives \
	    					+ false_positives + true_positives
	    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
	    precision = 1.0*true_positives/(true_positives+false_positives)
	    recall = 1.0*true_positives/(true_positives+false_negatives)
	    f1 = 2.0 * true_positives/(2*true_positives \
	    	+ false_positives+false_negatives)
	    result = []
	    if return_accuracy:
	    	result.append(accuracy)
	    if return_precision:
	    	result.append(precision)
	    if return_recall:
	    	result.append(recall)
	    if return_f1:
	    	result.append(f1)
	    return result
	except:
	    print "Got a divide by zero when trying out:", clf
	    print """
	    Precision or recall may be undefined due to 
	    a lack of true positive predicitons.
	    """


def plot_kbest_tree():
	"""Plot the number of selected features by their scores (DecisionTree)"""
	# anova filter
	transform = SelectKBest(f_classif)
	# Find the optimal number of features
	number_of_feature=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

	precision = []
	recall = []
	f1 = []

	for k in number_of_feature:

	    features_list=[]
	    features_list.append('poi')
	    clf = Pipeline([('anova', transform), ('tree', 
	    	tree.DecisionTreeClassifier())])
	    clf.set_params(anova__k=k).fit(features, labels)
	    selected =clf.named_steps['anova'].get_support()

	    for i in range(len(selected)):
	            if selected[i]== True:
	                features_list.append(features_considered[i+1])
	    
	    print features_list
	    
	    precision.append(get_score(clf, my_dataset, features_list, 
	    	return_precision = True)[0])
	    recall.append(get_score(clf, my_dataset, features_list, 
	    	return_recall = True)[0])
	    f1.append(get_score(clf, my_dataset, features_list, 
	    	return_f1 = True)[0])

	# Plot the Precision and Recall as a function of k of features
	plt.figure(figsize=(16,8))
	p1 =plt.plot(number_of_feature,precision,'r*-')
	p2 =plt.plot(number_of_feature,recall,'g*-')
	p3 = plt.plot(number_of_feature, f1, 'b--')
	plt.title( 'Precision and Recall and F-Score VS Number \
		of selected features (DecisionTree)')
	plt.xlabel('Number of selected features')
	plt.ylabel('Score')

	plt.legend((p1[0], p2[0], p3[0]), ( 'Precision','Recall', 'F-Score'))
	plt.show()

def plot_kbest_nb():
	"""Plot the number of selected features by their scores (GaussianNB)"""
	# anova filter
	transform = SelectKBest(f_classif)
	# Find the optimal number of features
	number_of_feature=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

	precision = []
	recall = []
	f1 = []

	for k in number_of_feature:

	    features_list=[]
	    features_list.append('poi')
	    clf = Pipeline([('anova', transform), 
			('GaussianNB', GaussianNB())])
	    clf.set_params(anova__k=k).fit(features, labels)
	    selected =clf.named_steps['anova'].get_support()

	    for i in range(len(selected)):
	            if selected[i]== True:
	                features_list.append(features_considered[i+1])
	    
	    print features_list
	    
	    precision.append(get_score(clf, my_dataset, features_list, 
	    	return_precision = True)[0])
	    recall.append(get_score(clf, my_dataset, features_list, 
	    	return_recall = True)[0])
	    f1.append(get_score(clf, my_dataset, features_list, 
	    	return_f1 = True)[0])

	# Plot the Precision and Recall as a function of k of features
	plt.figure(figsize=(16,8))
	p1 =plt.plot(number_of_feature,precision,'r*-')
	p2 =plt.plot(number_of_feature,recall,'g*-')
	p3 = plt.plot(number_of_feature, f1, 'b--')
	plt.title( 'Precision and Recall and F-Score VS Number \
		of selected features (GaussianNB)')
	plt.xlabel('Number of selected features')
	plt.ylabel('Score')

	plt.legend((p1[0], p2[0], p3[0]), ( 'Precision','Recall', 'F-Score'))
	plt.show()

def rfe():
	"""Recursive feature elimination"""
	model = LogisticRegression()
	# create the RFE model and select 3 attributes
	rfe = RFE(model, 3)
	rfe = rfe.fit(features_train, labels_train)
	# summarize the selection of the attributes
	print([features_considered[i + 1] for i in rfe.get_support(indices=True) ])
	print(rfe.ranking_)
	for i in range(len(rfe.ranking_)):
	    print features_considered[i+1], ": ", rfe.ranking_[i]



# plot_kbest_tree()
# plot_kbest_nb()
# rfe()
# al_tuning(features1)
#al_tuning(features2)
#al_tuning(features2)
