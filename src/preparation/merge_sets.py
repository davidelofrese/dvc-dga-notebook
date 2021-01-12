import pandas as pd

def merge_sets(alexa_set, dga_set):
	# Concatenate the domains
	merged = pd.concat([alexa_set, dga_set], ignore_index=True)
	return merged

if __name__ == '__main__':
	import argparse
	import os

	parser = argparse.ArgumentParser('merge_sets.py')
	parser.add_argument('alexa_set', help='Alexa set')
	parser.add_argument('dga_set', help='DGA set')
	parser.add_argument('set_type', help='Set type (training or test)')
	parser.add_argument('output_dir', help='Directory to save the merged set')
	args = parser.parse_args()

	# Load sets
	alexa_set = pd.read_pickle(args.alexa_set)
	dga_set = pd.read_pickle(args.dga_set)

	merged = merge_sets(alexa_set, dga_set)

	# Save merged set
	os.makedirs(args.output_dir, exist_ok=True)
	merged.to_pickle(os.path.join(args.output_dir, 'merged_{}_set.pkl'.format(args.set_type)))