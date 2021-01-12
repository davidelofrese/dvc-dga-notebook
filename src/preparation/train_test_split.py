def train_test_split(dataframe):
	# Hold out 10%
	total = dataframe.shape[0]
	hold_out_size = int(round(total*.9))
	hold_out = dataframe[hold_out_size:]	
	dataframe = dataframe[:hold_out_size]

	return { 'dataframe': dataframe, 'hold_out': hold_out }

if __name__ == '__main__':
	import argparse
	import pandas as pd
	import os

	parser = argparse.ArgumentParser('train_test_split.py')
	parser.add_argument('dataset', help='Prepared dataset')
	parser.add_argument('dataset_name', help='Dataset name')
	parser.add_argument('output_dir', help='Directory to save the splitted dataset')
	args = parser.parse_args()

	# Load dataset
	dataset = pd.read_pickle(args.dataset)

	splitted = train_test_split(dataset)

	# Save splitted training and test set
	os.makedirs(args.output_dir, exist_ok=True)
	splitted['dataframe'].to_pickle(os.path.join(args.output_dir, '{}_train.pkl'.format(args.dataset_name)))
	splitted['hold_out'].to_pickle(os.path.join(args.output_dir, '{}_test.pkl'.format(args.dataset_name)))