import pandas as pd
import tldextract
import numpy as np

def domain_extract(uri):
    ext = tldextract.extract(uri)
    if (not ext.suffix):
        return np.nan
    else:
        return ext.domain

def alexa_prepare(dataset, params):
	# Load Alexa dataset in a Pandas dataframe
	alexa_dataframe = pd.read_csv(dataset, names=['rank','uri'], header=None, encoding='utf-8')

	# For this exercise we need the 2LD and nothing else
	alexa_dataframe['domain'] = [ domain_extract(uri) for uri in alexa_dataframe['uri'] ]
	del alexa_dataframe['rank']
	del alexa_dataframe['uri']

	# It's possible we have NaNs from blanklines or whatever
	alexa_dataframe = alexa_dataframe.dropna()
	alexa_dataframe = alexa_dataframe.drop_duplicates()

	# Set the class
	alexa_dataframe['class'] = 'legit'

	# Shuffle the data (important for training/testing)
	np.random.seed(params['seed'])
	alexa_dataframe = alexa_dataframe.reindex(np.random.permutation(alexa_dataframe.index))

	return alexa_dataframe

if __name__ == '__main__':
	import argparse
	import yaml
	import os

	parser = argparse.ArgumentParser('alexa_prepare.py')
	parser.add_argument('dataset', help='Alexa dataset in CSV format')
	parser.add_argument('output_dir', help='Directory to save the prepared Alexa dataset')
	args = parser.parse_args()

	with open(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'params.yaml'), 'r') as pf:
		params = yaml.safe_load(pf)

	alexa_dataframe = alexa_prepare(args.dataset, params['preparation'])

	# Save prepared Alexa dataset
	os.makedirs(args.output_dir, exist_ok=True)
	alexa_dataframe.to_pickle(os.path.join(args.output_dir, 'alexa_prepared.pkl'))