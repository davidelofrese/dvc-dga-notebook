import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model(all_domains, params):
	# We exclude the weird class from our ML training
	not_weird = all_domains[all_domains['class'] != 'weird']

	# List of feature vectors (scikit learn uses 'X' for the matrix of feature vectors)
	X = not_weird[['length', 'entropy', 'alexa_grams', 'word_grams']].values

	# Labels (scikit learn uses 'y' for classification labels)
	y = np.array(not_weird['class'].tolist())

	# Random Forest is a popular ensemble machine learning classifier.
	# http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html
	clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['seed'])
	
	# Fit the random forest model
	clf.fit(X, y)

	return clf

if __name__ == '__main__':
	import argparse
	import pickle
	import yaml
	import os
	import pandas as pd

	parser = argparse.ArgumentParser('train_model.py')
	parser.add_argument('training_set', help='Training set')
	parser.add_argument('output_dir', help='Directory to save the trained model')
	args = parser.parse_args()

	with open(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'params.yaml'), 'r') as pf:
		params = yaml.safe_load(pf)	

	# Load the training set
	training_set = pd.read_pickle(args.training_set)

	clf = train_model(training_set, params['models'])

	# Save trained model
	os.makedirs(args.output_dir, exist_ok=True)
	pickle.dump(clf, open(os.path.join(args.output_dir, 'trained_model.pkl'), 'wb'))