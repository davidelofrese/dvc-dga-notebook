import pandas as pd

def dga_prepare(dataset):
	# Read in the DGA domains
	dga_dataframe = pd.read_csv(dataset, names=['raw_domain'], header=None, encoding='utf-8')

	# The blacklist values just differ by capitalization or .com/.org/.info
	dga_dataframe['domain'] = dga_dataframe.applymap(lambda x: x.split('.')[0].strip().lower())
	del dga_dataframe['raw_domain']

	# It's possible we have NaNs from blanklines or whatever
	dga_dataframe = dga_dataframe.dropna()
	dga_dataframe = dga_dataframe.drop_duplicates()

	# Set the class
	dga_dataframe['class'] = 'dga'

	return dga_dataframe

if __name__ == '__main__':
	import argparse
	import os

	parser = argparse.ArgumentParser('dga_prepare.py')
	parser.add_argument('dataset', help='DGA domains dataset in TXT format')
	parser.add_argument('output_dir', help='Directory to save the prepared DGA domains dataset')
	args = parser.parse_args()

	dga_dataframe = dga_prepare(args.dataset)

	# Save prepared DGA domains dataset
	os.makedirs(args.output_dir, exist_ok=True)
	dga_dataframe.to_pickle(os.path.join(args.output_dir, 'dga_prepared.pkl'))