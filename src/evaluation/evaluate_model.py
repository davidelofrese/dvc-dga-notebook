def evaluate_model(clf, hold_out_domains):
	# List of feature vectors
	hold_X = hold_out_domains[['length', 'entropy', 'alexa_grams', 'word_grams']].values

	# Run through the predictive model
	hold_y_pred = clf.predict(hold_X)

	# Add the prediction array to the dataframe
	hold_out_domains['pred'] = hold_y_pred

	return hold_out_domains

if __name__ == '__main__':
	import argparse
	import os
	import pickle
	import pandas as pd

	parser = argparse.ArgumentParser('evaluate_model.py')
	parser.add_argument('model', help='Trained model')
	parser.add_argument('test_set', help='Test set')
	parser.add_argument('output_dir', help='Directory to save the results')
	args = parser.parse_args()

	# Load trained model and test set
	model = pickle.load(open(args.model, 'rb'))
	test_set = pd.read_pickle(args.test_set)

	hold_out_domains = evaluate_model(model, test_set)

	# Save results
	os.makedirs(args.output_dir, exist_ok=True)
	hold_out_domains[['class', 'pred']].to_csv(open(os.path.join(args.output_dir, 'classes.csv'), 'w'), index=False)