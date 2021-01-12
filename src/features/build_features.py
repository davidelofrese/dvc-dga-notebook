import numpy as np
import math
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Grabbed this from Rosetta Code (rosettacode.org) 
def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def vectorize(dataframe, column, params):
	# Scikit learn has a NGram generator that can generate either char NGrams or word NGrams (we're using char).
	vc = CountVectorizer(analyzer='char', ngram_range=(params['range_low'], params['range_high']), min_df=params['min_df'], max_df=params['max_df'])
	counts_matrix = vc.fit_transform(dataframe[column])
	counts = np.log10(counts_matrix.sum(axis=0).getA1())

	return {'vc': vc, 'counts': counts}

def build_features(all_domains, hold_out_domains, alexa_prepared, words_prepared, params):
	# Copy the dataframes to avoid SettingWithCopyWarning
	all_domains = all_domains.copy()
	hold_out_domains = hold_out_domains.copy()

	# Add a length field for the domain
	all_domains['length'] = [len(x) for x in all_domains['domain']]
	hold_out_domains['length'] = [len(x) for x in hold_out_domains['domain']]

	# Since we're trying to detect dynamically generated domains and short
	# domains (length <= 6) are crazy random even for 'legit' domains we're going
	# to punt on short domains
	all_domains = all_domains[all_domains['length'] > 6]
	hold_out_domains = hold_out_domains[hold_out_domains['length'] > 6]

	# Add a entropy field for the domain
	all_domains['entropy'] = [entropy(x) for x in all_domains['domain']]
	hold_out_domains['entropy'] = [entropy(x) for x in hold_out_domains['domain']]

	# Vectorize prepared datasets
	alexa_vc = vectorize(alexa_prepared, 'domain', params['alexa_vectorization'])
	words_vc = vectorize(words_prepared, 'word', params['words_vectorization'])

	# Compute NGram matches for all the domains and add to our dataframe
	all_domains['alexa_grams'] = alexa_vc['counts'] * alexa_vc['vc'].transform(all_domains['domain']).T 
	all_domains['word_grams'] = words_vc['counts'] * words_vc['vc'].transform(all_domains['domain']).T
	hold_out_domains['alexa_grams'] = alexa_vc['counts'] * alexa_vc['vc'].transform(hold_out_domains['domain']).T
	hold_out_domains['word_grams'] = words_vc['counts'] * words_vc['vc'].transform(hold_out_domains['domain']).T

	# Legit domains which are scoring low on both alexa and word gram count
	weird_cond = (all_domains['class'] == 'legit') & (all_domains['word_grams'] < 3) & (all_domains['alexa_grams'] < 2)
	all_domains.loc[weird_cond, 'class'] = 'weird'

	return { 'all_domains': all_domains, 'hold_out_domains': hold_out_domains }

if __name__ == '__main__':
	import argparse
	import pandas as pd
	import yaml
	import os

	parser = argparse.ArgumentParser('build_features.py')
	parser.add_argument('merged_training_set', help='Merged training set')
	parser.add_argument('merged_test_set', help='Merged test set')
	parser.add_argument('alexa_prepared', help='Alexa prepared dataset')
	parser.add_argument('words_prepared', help='Words prepared dataset')
	parser.add_argument('output_dir', help='Directory to save the datasets')
	args = parser.parse_args()

	with open(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'params.yaml'), 'r') as pf:
		params = yaml.safe_load(pf)

	# Read in the merged datasets
	merged_training_set = pd.read_pickle(args.merged_training_set)
	merged_test_set = pd.read_pickle(args.merged_test_set)

	# Read in the prepared datasets
	alexa_prepared = pd.read_pickle(args.alexa_prepared)
	words_prepared = pd.read_pickle(args.words_prepared)

	features = build_features(merged_training_set, merged_test_set, alexa_prepared, words_prepared, params['features'])

	# Save training and test set
	os.makedirs(args.output_dir, exist_ok=True)
	features['all_domains'].to_pickle(os.path.join(args.output_dir, 'training_set.pkl'))
	features['hold_out_domains'].to_pickle(os.path.join(args.output_dir, 'test_set.pkl'))