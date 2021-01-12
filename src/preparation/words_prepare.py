import pandas as pd

def words_prepare(dataset):
	# Read in the dictionary words
	word_dataframe = pd.read_csv(dataset, names=['word'], header=None, dtype={'word': str}, encoding='utf-8')

	# Cleanup words from dictionary
	word_dataframe = word_dataframe[word_dataframe['word'].map(lambda x: str(x).isalpha())]
	word_dataframe = word_dataframe.applymap(lambda x: str(x).strip().lower())
	word_dataframe = word_dataframe.dropna()
	word_dataframe = word_dataframe.drop_duplicates()

	return word_dataframe

if __name__ == '__main__':
	import argparse
	import os
	
	parser = argparse.ArgumentParser('words_prepare.py')
	parser.add_argument('dataset', help='Words dataset in TXT format')
	parser.add_argument('output_dir', help='Directory to save the prepared words dataset')
	args = parser.parse_args()

	word_dataframe = words_prepare(args.dataset)

	# Save prepared words dataset
	os.makedirs(args.output_dir, exist_ok=True)
	word_dataframe.to_pickle(os.path.join(args.output_dir, 'words_prepared.pkl'))